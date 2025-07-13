import cupy as cp
from cupy.random import _kernels as _rk

# ----------------------- preamble (ASCII only) ------------------------
preamble = (
    "#define M_PI 3.14159265358979323846\n"
    + _rk.rk_basic_definition
    + _rk.loggam_definition
    + _rk.long_min_max_definition
    + _rk.rk_hypergeometric_definition
)

kernel_src = r'''
// ── helpers ----------------------------------------------------------
__device__ inline float logC(int n,int r){
    return loggam(n+1) - loggam(r+1) - loggam(n-r+1);
}
__device__ inline float logP(int w,int b,int k,int x){
    return logC(w,x) + logC(b,k-x);
}
// --------------------------------------------------------------------
extern "C" __global__
void ffh_mh_seq_flat(
    const int *W,       // D*B flattened whites
    const int *Bk,      // D*B flattened blacks
    const int *K,       // D*B flattened draws
    const int *S,       // D   target per table
    const int  Bdim,    // number of bins = B
    const int  sweeps,  // MH sweeps
    const unsigned long long seed,
    int       *X        // D*B flattened output
){
    int d = blockIdx.x;
    if (threadIdx.x != 0) return;

    int base = d * Bdim;
    rk_state st; rk_seed(seed + (unsigned long long)d, &st);

    // 1) independent HG draws
    int sum = 0;
    for (int i = 0; i < Bdim; ++i) {
        int wi = W[base + i];
        int bi = Bk[ base + i];
        int ki = K[  base + i];
        int xi = rk_hypergeometric(&st, wi, bi, ki);
        X[base + i] = xi;
        sum += xi;
    }

    // 2) repair to per-bin rounded mean
    int below[128], above[128];
    int gapB[128], gapA[128];
    int nB = 0, nA = 0;
    int totB = 0, totA = 0;
    for (int i = 0; i < Bdim; ++i) {
        int wi = W[base + i];
        int bi = Bk[ base + i];
        int ki = K[  base + i];
        // rounded mean m_i = (k*w + (w+b)/2) / (w+b)
        int m = (int)(((long long)ki * wi + (wi+bi)/2) / (wi+bi));
        int xi= X[base + i];
        int gi= m - xi;
        if (gi > 0) { below[nB] = i; gapB[nB] = gi; totB += gi; ++nB; }
        else if (gi < 0) { above[nA] = i; gapA[nA] = -gi; totA += -gi; ++nA; }
    }
    int diff = S[d] - sum;

    // add whites
    if (diff > 0) {
        int rem = diff;
        for (int idx = 0; idx < nB && rem > 0; ++idx) {
            int i    = below[idx];
            int hi   = min(W[base+i], K[base+i]);
            int head = hi - X[base+i];
            int share= (int)(((long long)rem * gapB[idx]) / totB);
            if (share > head) share = head;
            X[base+i] += share;
            rem       -= share;
        }
        for (int idx = 0; idx < nB && rem > 0; ++idx) {
            int i = below[idx];
            if (X[base+i] < min(W[base+i], K[base+i])) {
                X[base+i]++; --rem;
            }
        }
    }
    // remove whites
    else if (diff < 0) {
        int rem = -diff;
        for (int idx = 0; idx < nA && rem > 0; ++idx) {
            int i    = above[idx];
            int lo   = max(0, K[base+i] - Bk[base+i]);
            int room = X[base+i] - lo;
            int share= (int)(((long long)rem * gapA[idx]) / totA);
            if (share > room) share = room;
            X[base+i] -= share;
            rem       -= share;
        }
        for (int idx = 0; idx < nA && rem > 0; ++idx) {
            int i = above[idx];
            int lo = max(0, K[base+i] - Bk[base+i]);
            if (X[base+i] > lo) {
                X[base+i]--; --rem;
            }
        }
    }

    // ----- Before MH: build donor (dec) & receiver (inc) lists -----
    int incList[128], decList[128];
    int incN = 0, decN = 0;

    for (int i = 0; i < Bdim; ++i) {
        int lo = max(0,   K[base+i] - Bk[base+i]);
        int hi = min(W[base+i], K[base+i]);
        int xi = X[base+i];
        if (xi > lo) decList[decN++] = i;
        if (xi < hi) incList[incN++] = i;
    }

    // ----- 3) MH token‐swap using only valid bins -----
    for (int s = 0; s < sweeps; ++s) {
        if (decN == 0 || incN == 0) break;  // nothing to swap

        // pick donor index d and receiver index r
        int di = rk_random(&st) % decN;
        int ri = rk_random(&st) % incN;
        int i  = decList[di];
        int j  = incList[ri];
        if (i == j) continue;

        // load values
        int xi = X[base+i], xj = X[base+j];
        int wi = W[base+i], bi = Bk[base+i], ki = K[base+i];
        int wj = W[base+j], bj = Bk[base+j], kj = K[base+j];

        // bounds
        int lo_i = max(0,   ki - bi), hi_i = min(wi, ki);
        int lo_j = max(0,   kj - bj), hi_j = min(wj, kj);

        // propose move: take one white from i → j
        int xi2 = xi - 1;
        int xj2 = xj + 1;

        // compute log‐ratio
        float lr = logP(wi,bi,ki,xi2)
                + logP(wj,bj,kj,xj2)
                - logP(wi,bi,ki, xi )
                - logP(wj,bj,kj, xj );

        // single U for accept
        float u = rk_double(&st);
        // printf("u: %f, lr: %f, exp(lr): %f\n", u, lr, expf(lr));
        if (u < expf(lr)) {
            // printf("accepting swap\n");
            // accept
            X[base+i] = xi2;
            X[base+j] = xj2;

            // --- update decList for i ---
            if (xi2 == lo_i) {
                // no longer can donate
                decList[di] = decList[--decN];
            }
            // if was at hi_i, now can receive → add to incList
            if (xi  == hi_i) {
                incList[incN++] = i;
            }

            // --- update incList for j ---
            if (xj2 == hi_j) {
                // no longer can receive
                incList[ri] = incList[--incN];
            }
            // if was at lo_j, now can donate → add to decList
            if (xj  == lo_j) {
                decList[decN++] = j;
            }
        }
    }

}
'''

ffh_mh_flat = cp.RawKernel(preamble + kernel_src, 'ffh_mh_seq_flat')

def ffh(w, b, k, S, sweeps=48, seed=None):
    """
    w, b, k:  (D,B) arrays  (whites, blacks, draws)
    S     :  (D,)        (target totals, <=100)
    """
    Wf = cp.asarray(w, dtype=cp.int32).ravel()
    Bf = cp.asarray(b, dtype=cp.int32).ravel()
    Kf = cp.asarray(k, dtype=cp.int32).ravel()
    Sf = cp.asarray(S, dtype=cp.int32)
    D, Bdim = w.shape
    Xf = cp.empty_like(Wf)
    seed = int(seed or cp.random.randint(0,2**63-1,(),dtype=cp.uint64))
    ffh_mh_flat((D,), (1,),
                (Wf, Bf, Kf, Sf, Bdim, sweeps, seed, Xf))
    cp.cuda.Stream.null.synchronize()
    return Xf.reshape(D, Bdim)
