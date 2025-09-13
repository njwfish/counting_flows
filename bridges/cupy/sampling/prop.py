import cupy as cp
import numpy as np
from .multinomial import multinomial
from .mvhg import mvhg

from ..utils import dlpack_backend

def proportional_proj(x, target):
    """
    Take in a (possibly batched) source matrix of integers and target vector and project the source to match the column sums of the target.

    Args:
        x: (possibly batched) source matrix of integers
        target: target sum vector

    Returns:
        projected source matrix of integers
    """
    if not isinstance(x, cp.ndarray):
        x, target = dlpack_backend(x, target, backend="cupy")

    batched = len(x.shape) > 2 or len(target.shape) > 1
    if len(x.shape) > 2 and len(target.shape) > 1:
        target_b, target_d = target.shape
        b, n, d = x.shape
        assert target_b == b and target_d == d
        target = target.flatten() # [b * d]
        # tranpose to [n, b, d] and then flatten to [n, b * d]
        x = x.transpose(1, 0, 2).reshape(n, b * d)
    elif len(target.shape) > 1:
        target_b, target_d = target.shape
        m, d = x.shape
        assert m % target_b == 0
        n, b = m // target_b, target_b
        target = target.flatten() # [b * d]
        # tranpose to [n, b, d] and then flatten to [n, b * d]
        x = x.reshape(b, n, d).transpose(1, 0, 2).reshape(n, b * d)
    else:
        target_d = target.shape[0]
        n, d = x.shape
        assert target_d == d

    # this code assumes that the target and x are non-negative
    assert (target >= 0).all()
    assert (x >= 0).all()


    x = x.astype(cp.int32)
    curr = x.sum(axis=0)
    target = target.astype(cp.int32)

    diff = target - curr
    delta = cp.zeros_like(x)
    
    diff_pos = diff > 0
    if cp.any(diff_pos):
        # if the diff is positive target is larger than current and we need to add counts   
        curr_neg_non_zero = curr[diff_pos] > 0
        # we could add a smarter prior here where we sample proportional to the current counts per cell instead of uniformly
        # this would involve multiplying by like x.sum(axis=0) instead of just doing ones_like
        p = cp.ones_like(x[:, diff_pos], dtype=cp.float32) 
        p[:, curr_neg_non_zero] = x[:, diff_pos][:, curr_neg_non_zero].astype(cp.float32) / curr[diff_pos][curr_neg_non_zero] # .reshape(1)
        samp = multinomial(n=diff[diff_pos], p=p)
        delta[:, diff_pos] = samp
    
    diff_neg = diff < 0
    if cp.any(diff_neg):
        # if the diff is positive target is smaller than current and we need to remove counts
        # we can do this via our multivariate hypergeometric
        curr_pos_non_zero = curr[diff_neg] > 0
        delta[:, diff_neg] = -mvhg(
            pop=x[:, diff_neg],
            draws_tot=-diff[diff_neg]
        )

    x = x + delta
    
    if batched:
        x = x.reshape(n, b, d).transpose(1, 0, 2)

    return x
    