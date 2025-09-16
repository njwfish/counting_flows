import cupy as cp

def multinomial(
    n: cp.ndarray,
    p: cp.ndarray
) -> cp.ndarray:
    """
    Draw samples from a Multinomial for *different* total-counts and probs
    in one batch.  Internally does K sequential Binomial draws.

    Args:
      n: ndarray of shape [...], nonnegative integers (total count per batch entry)
      p: ndarray of shape [K, ...], nonnegative, will be renormalized along last axis

    Returns:
      counts: ndarray of shape [K, ...], summing (per‑batch) to the respective n.
    """
    # ensure p is a proper probability vector per batch
    # remove zeros to avoid division by zero
    # p = p / p.sum(axis=-1, keepdims=True)

    # broadcast n up to the batch shape
    batch_shape = p.shape[1:]
    n = cp.broadcast_to(n, batch_shape)

    # compute the “remaining probability” at each category:
    # reversed_cumsum[..., i] = sum_{j=i..K-1} p[..., j]
    reversed_cumsum = cp.flip(cp.flip(p, axis=0).cumsum(axis=0), axis=0)

    # ratio for class i is p[i] / remaining_cumsum[i]
    ratios = p / reversed_cumsum.clip(min=1e-8)

    remaining = n.astype(cp.float32)  # for Binomial’s total_count argument
    counts = cp.zeros_like(p, dtype=cp.int32)
    # loop over K categories
    for i in range(p.shape[0]):
        r = ratios[i, ...]
        # sample how many of the remaining trials go to class i
        counts[i, ...] = cp.random.binomial(n=remaining, p=r, dtype=cp.int32)
        remaining = remaining - counts[i, ...]
    
    return counts