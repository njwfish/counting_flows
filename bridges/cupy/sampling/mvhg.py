# correct

import cupy as cp
from cupy.random import _kernels as _rk

# Assemble ASCII‐only preamble
preamble = (
    "#define M_PI 3.14159265358979323846\n"
  + _rk.rk_basic_definition
  + _rk.loggam_definition
  + _rk.long_min_max_definition
  + _rk.rk_hypergeometric_definition
)

kernel_one_fixed = r'''
extern "C" __global__
void mvhg_onecolour_seq(
    const int * __restrict__ pop_flat,   // D*B
    const int   B,                       // bins per row
    const int * __restrict__ draws_tot,  // D total draws
    const unsigned long long seed,       // RNG seed
    int * __restrict__ out_flat          // D*B output
){
    int d = blockIdx.x;
    if (threadIdx.x != 0) return;

    // RNG init
    rk_state st;
    rk_seed(seed + (unsigned long long)d, &st);

    const int *pop_row = pop_flat + d*B;
          int *out_row = out_flat + d*B;

    // compute total population
    int pop_rem   = 0;
    for (int i = 0; i < B; ++i) pop_rem += pop_row[i];
    int rem_draws = draws_tot[d];

    // sequentially split
    for (int i = 0; i < B-1; ++i) {
        int Ni = pop_row[i];
        int draw = 0;
        if (rem_draws > 0) {
            // only call when there's something to draw
            draw = rk_hypergeometric(
                       &st,
                       Ni,            // ngood
                       pop_rem - Ni,  // nbad
                       rem_draws      // nsample
                   );
        }
        out_row[i]   = draw;
        pop_rem     -= Ni;
        rem_draws   -= draw;
    }
    // last bin gets any remaining draws
    out_row[B-1] = rem_draws;
}
'''
split_onecolour_seq = cp.RawKernel(preamble + kernel_one_fixed,
                                   'mvhg_onecolour_seq')

def mvhg(pop, draws_tot):
    """Exact GPU MVHG for jump counts, safe against zero-sample."""
    pop32   = pop.astype(cp.int32)
    draws32 = draws_tot.astype(cp.int32)
    D, B    = pop32.shape
    out     = cp.empty((D, B), dtype=cp.int32)
    seed    = int(cp.random.randint(0, 2**63-1, (), dtype=cp.uint64).item())

    split_onecolour_seq(
        (D,), (1,),
        (pop32.ravel(), B, draws32, seed, out.ravel())
    )
    # cp.cuda.Stream.null.synchronize()
    return out

