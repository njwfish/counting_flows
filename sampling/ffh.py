import numpy as np
from scipy import special
from typing import Union, Tuple, Optional
import time

import numpy as np
from scipy import special



def hypergeom_logpmf_numpy(k, N, K, n):
    """
    Numerically robust log-pmf of the hypergeometric distribution.

    – Broadcasts all four arguments to a common shape.
    – Returns –inf outside the support instead of NaN / +/-inf.
    """
    logp = (
        special.gammaln(K + 1)          - special.gammaln(k + 1) - special.gammaln(K - k + 1)
        + special.gammaln(N - K + 1)     - special.gammaln(n - k + 1)
        - special.gammaln(N - K - n + k + 1)
        - (special.gammaln(N + 1) - special.gammaln(n + 1) - special.gammaln(N - n + 1))
    )

    return logp


def mh_mean_constrained_update_numpy(
    N_t: np.ndarray,     # (B,d) total population at time-t
    B_t: np.ndarray,     # (B,d) total successes at time-t
    N_s: np.ndarray,     # (B,d) sample size at time-s
    B_s: np.ndarray,     # (B,d) sample successes at time-s  ← **this** is both input & output
    S:   np.ndarray,     # (d,) target *mean* per column
    sweeps: int = 0,
    max_projections: int = 5,
    override_support: bool = False,
) -> np.ndarray:
    """
    MH update on B_s so that:
      • each entry stays in [0, min(N_s, B_t)],
      • each column‐sum = round(B * M[j]),
      • the hypergeometric target stays invariant.
    """
    B, d = B_s.shape
    # work in-place on a copy
    B_s = B_s.copy()
    lower = np.maximum(0, N_s + B_t - N_t)   # min successes possible
    upper = np.minimum(N_s, B_t)            # max successes possible
    
    # 0) quick projections to hit ∑_i B_s[i,j] = S_target[j]
    for _ in range(max_projections):
        B_s, _ = constrained_multinomial_proposal_numpy(
            N_t, B_t, N_s, B_s, S, override_support=False
        )
        # print("B_s", B_s.sum(axis=0), S)
        if np.all(B_s.sum(axis=0) == S):
            break

    # 1) MH sweeps via ±1 exchanges in each column
    rng  = np.random.default_rng()

    for _ in range(sweeps):
        # 1) feasibility masks
        can_inc = (B_s < upper)   # shape (B,d)
        can_dec = (B_s > lower)   # shape (B,d)

        # 2) how many rows we can inc/dec in each column
        cnt_inc = can_inc.sum(axis=0)  # (d,)
        cnt_dec = can_dec.sum(axis=0)  # (d,)

        # 3) which columns are even eligible?
        good = (cnt_inc > 0) & (cnt_dec > 0)  # (d,)
        if not good.any():
            continue

        cols = np.nonzero(good)[0]           # list of good j's

        # 4) sample a random "rank" in each good column
        r_inc = rng.integers(0, cnt_inc[cols])   # how many trues to skip
        r_dec = rng.integers(0, cnt_dec[cols])

        # 5) cumsum trick: find the row where cumsum == rank+1
        #    restrict to only the good columns
        sub_inc = can_inc[:, cols]           # (B, n_good)
        sub_dec = can_dec[:, cols]

        csum_inc = np.cumsum(sub_inc, axis=0)  # (B, n_good)
        csum_dec = np.cumsum(sub_dec, axis=0)

        # pick p_rows and q_rows
        # we look for the first row where csum == r+1
        target_inc = r_inc[np.newaxis, :] + 1   # broadcast to (1,n_good)
        target_dec = r_dec[np.newaxis, :] + 1

        # this yields a boolean array (B, n_good), only one True per column
        pick_inc = (csum_inc == target_inc) & sub_inc
        pick_dec = (csum_dec == target_dec) & sub_dec

        # and argmax gives the row index
        p_rows = np.argmax(pick_inc, axis=0)  # shape (n_good,)
        q_rows = np.argmax(pick_dec, axis=0)

        # make sure p != q (if they collide, just skip that column)
        neq = (p_rows != q_rows)
        if not neq.any():
            continue

        sel     = np.nonzero(neq)[0]
        j_idx   = cols[sel]
        p_idx   = p_rows[sel]
        q_idx   = q_rows[sel]

        # 6) compute current/proposed counts
        curr_p = B_s[p_idx, j_idx]
        curr_q = B_s[q_idx, j_idx]
        prop_p = curr_p + 1
        prop_q = curr_q - 1

        # --- 6b) Re-filter to make sure prop_p, prop_q are in [lower,upper] ---
        p_lo = lower[p_idx, j_idx]
        p_hi = upper[p_idx, j_idx]
        q_lo = lower[q_idx, j_idx]
        q_hi = upper[q_idx, j_idx]

        support_ok = (
            (prop_p >= p_lo) & (prop_p <= p_hi) &
            (prop_q >= q_lo) & (prop_q <= q_hi)
        )
        if not support_ok.any():
            continue  # nothing valid this sweep

        # Select only the truly valid proposals
        sel        = support_ok.nonzero()[0]
        j_sel      = j_idx[sel]
        p_sel      = p_idx[sel]
        q_sel      = q_idx[sel]
        curr_p     = curr_p[sel]
        curr_q     = curr_q[sel]
        prop_p     = prop_p[sel]
        prop_q     = prop_q[sel]

        # --- 7) compute log-densities only on valid entries ---
        logp_curr = (
            hypergeom_logpmf_numpy(curr_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_numpy(curr_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )
        logp_prop = (
            hypergeom_logpmf_numpy(prop_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_numpy(prop_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )

        # --- 8) Clip and cap the MH ratio ---
        dlog  = logp_prop - logp_curr
        dlog  = np.clip(dlog, -50, +50)        # avoid overflow in exp
        ratio = np.exp(dlog)
        ratio = np.minimum(ratio, 1.0)         # α = min(1, ratio)


        accept = rng.random(len(ratio)) < ratio

        # print("numpy", time.time() - time_start, "logp_curr", logp_curr.mean(),"logp_prop", logp_prop.mean(), "ratio", ratio.mean(), "accept", accept.mean())
        if not accept.any():
            continue

        # --- 9) Apply accepted moves ---
        a_sel = sel[accept]
        j_acc = j_idx[a_sel]
        p_acc = p_idx[a_sel]
        q_acc = q_idx[a_sel]

        B_s[p_acc, j_acc] += 1
        B_s[q_acc, j_acc] -= 1

    return B_s

def constrained_multinomial_proposal_numpy(
    N_t: np.ndarray,     # (B,d)
    B_t: np.ndarray,     # (B,d)
    N_s: np.ndarray,     # (B,d)
    B_s: np.ndarray,     # (B,d)
    S:   np.ndarray,     # (d,)
    override_support: bool = False,
) -> Tuple[np.ndarray,np.ndarray]:
    B, d = B_s.shape

    # 1) get true hyper‐geom support per‐entry
    lower = np.maximum(0,  N_s + B_t - N_t)   # (B,d)
    upper = np.minimum(N_s, B_t)              # (B,d)

    # 2) expected under hypergeom
    B_s_exp = np.where(N_t==0, 0.0, N_s * (B_t/N_t))

    # 3) column residuals & sign
    col_sum = B_s.sum(axis=0)     # (d,)
    R       = S - col_sum        # (d,)
    sgn     = np.sign(R).astype(int)  # (d,)
    Rabs    = np.abs(R).astype(int)   # (d,)

    # print("B_s_exp - S", np.sort(B_s_exp.sum(axis=0) - S))

    # 4) directional weights → probs
    #    we want to favor moves that shrink |B_s - E[B_s]|
    diff    = (B_s_exp - B_s) * sgn[np.newaxis,:]   # (B,d)
    w       = np.logaddexp(0, diff)                    # (B,d)
    probs   = w / w.sum(axis=0, keepdims=True)      # (B,d)

    # 5) vectorized multinomial for each column j
    rng     = np.random.default_rng()
    # transpose so each row is a p‐vector for one column
    p_mat   = probs.T       # shape (d,B)
    # pass array of counts Rabs (length d) and 2D p_mat → returns (d,B)
    draws   = rng.multinomial(Rabs, p_mat)
    # back to (B,d)
    delta   = draws.T       # (B,d)

    # 6) raw proposal and clip to true support
    B_prop = B_s + sgn[np.newaxis,:] * delta
    if not override_support:
        B_prop = np.clip(B_prop, lower, upper)

    return B_prop, delta
