import torch
from torch.distributions import Binomial

def multinomial(
    n: torch.Tensor,
    p: torch.Tensor
) -> torch.Tensor:
    """
    Draw samples from a Multinomial for *different* total-counts and probs
    in one batch.  Internally does K sequential Binomial draws.

    Args:
      n: Tensor of shape [...], nonnegative integers (total count per batch entry)
      p: Tensor of shape [..., K], nonnegative, will be renormalized along last axis

    Returns:
      counts: LongTensor of shape [..., K], summing (per‑batch) to the respective n.
    """
    # ensure p is a proper probability vector per batch
    p = p / p.sum(dim=-1, keepdim=True)

    # broadcast n up to the batch shape
    batch_shape = p.shape[:-1]
    n = n.expand(batch_shape)

    # compute the “remaining probability” at each category:
    # reversed_cumsum[..., i] = sum_{j=i..K-1} p[..., j]
    reversed_cumsum = p.flip(-1).cumsum(-1).flip(-1)

    # ratio for class i is p[i] / remaining_cumsum[i]
    ratios = p / reversed_cumsum.clamp(min=1e-8)

    remaining = n.float()  # for Binomial’s total_count argument
    counts = torch.zeros_like(p, dtype=torch.long)
    # loop over K categories
    for i in range(p.size(-1)):
        r = ratios[..., i]
        # sample how many of the remaining trials go to class i
        counts[..., i] = Binomial(total_count=remaining, probs=r).sample()
        remaining = remaining - counts[..., i]
    
    return counts
