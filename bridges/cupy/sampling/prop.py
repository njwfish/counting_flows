import cupy as cp
from .multinomial import multinomial
from .mvhg import mvhg

def proportional_proj(x, target):
    x = x.astype(cp.int32)
    curr = x.sum(axis=1)
    target = target.astype(cp.int32)

    diff = target - curr
    delta = cp.zeros_like(x)
    
    diff_neg = diff > 0
    if cp.any(diff_neg):
        # if the diff is negative target is larger than current and we need to add counts   
        curr_neg_non_zero = curr[diff_neg] > 0
        p = cp.zeros_like(x[diff_neg, :], dtype=cp.float32)
        p[curr_neg_non_zero, :] = x[diff_neg, :][curr_neg_non_zero, :].astype(cp.float32) / curr[diff_neg][curr_neg_non_zero].reshape(-1, 1)
        delta[diff_neg, :] = multinomial(n=diff[diff_neg], p=p)
    
    diff_pos = diff < 0
    if cp.any(diff_pos):
        # if the diff is positive target is smaller than current and we need to remove counts
        # we can do this via our multivariate hypergeometric
        curr_pos_non_zero = curr[diff_pos] > 0
        delta[diff_pos, :] = -mvhg(
            pop=x[diff_pos, :],
            draws_tot=-diff[diff_pos]
        )

    return x + delta
    