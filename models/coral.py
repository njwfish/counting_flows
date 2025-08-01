import torch
import torch.nn as nn
import torch.nn.functional as F

class CORALLoss(nn.Module):
    """
    Ordinal (CORAL) loss for integer‑valued targets.

    The base `architecture` must output logits of shape
        (B, D, K)  with  K = value_range     ( = V‑1 thresholds )

    Each logit s_k models  P[Y > k]  via  sigmoid(s_k).

    Parameters
    ----------
    architecture : nn.Module
        Called as  architecture(**inputs)  and expected to return (B,D,K) logits.
    min_value : int
        Smallest integer label.
    value_range : int
        (max_label - min_label).  If your counts run 0..127, set value_range=127.
    """
    def __init__(self, architecture, *, min_value: int = 0, value_range: int = 127):
        super().__init__()
        self.arch = architecture
        self.min_value   = min_value
        self.value_range = value_range            # ⇒ V = value_range + 1 labels
        self.K           = value_range            # thresholds k = 0 … K‑1
        print(f"[CORAL] support = [{min_value} … {min_value + value_range}] "
              f"(K = {self.K} thresholds)")

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _hazard(self, logits):
        """
        Convert logits → hazard probabilities  h_k = P[Y > k].

        Returns tensor (B,D,K) in (0,1).  We *clip* to [eps, 1‑eps] so that the
        PMF derived later is non‑negative even if the monotonicity is imperfect.
        """
        eps = 1e-6
        return torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    def _pmf_from_hazard(self, h):
        """
        Given hazard probs h_k = P[Y > k], derive a PMF over the V classes.

        pmf_0        = 1 - h_0
        pmf_k        = h_{k-1} - h_k                for 1 ≤ k ≤ V‑2
        pmf_{V-1}    = h_{K-1}
        """
        # pad: h_left[0] = 1   and   h_right[last] = 0
        h_left  = torch.cat([torch.ones_like(h[..., :1]), h], dim=-1)       # (..., K+1)
        h_right = torch.cat([h, torch.zeros_like(h[..., :1])], dim=-1)      # (..., K+1)

        pmf = h_left - h_right                                              # (..., V)
        pmf = pmf.clamp_min(0.0)                                            # avoid tiny negs
        pmf = pmf / pmf.sum(dim=-1, keepdim=True)                           # renormalise
        return pmf

    # --------------------------------------------------------------------- #
    # API
    # --------------------------------------------------------------------- #
    def forward(self, inputs):
        """
        Forward pass through the user architecture.  Returns raw logits
        with shape (B, D, K).  Nothing else happens here.
        """
        return self.arch(**inputs)

    def loss(self, target, inputs):
        """
        Batch‑mean CORAL loss (sum of binary cross‑entropies over thresholds).

        target : (B, D) integer tensor    in  [min_value, min_value+value_range]
        inputs : dict forwarded to the base architecture
        """
        logits = self.forward(inputs)                         # (B,D,K)
        if logits.size(-1) != self.K:
            raise ValueError(f"architecture must output K={self.K} logits per bin "
                             f"(got {logits.size(-1)})")

        # shift targets to 0 … V‑1
        y = (target - self.min_value).long().clamp(0, self.value_range)     # (B,D)

        # build binary targets  t_{k} = 1{ y > k }
        thresholds = torch.arange(self.K, device=logits.device)             # (K,)
        T = (y.unsqueeze(-1) > thresholds).float()                          # (B,D,K)

        bce = F.binary_cross_entropy_with_logits(logits, T, reduction='none')
        return 10 * bce.mean()

    def sample(self, **inputs):
        """
        Draw integer samples from the CORAL model.

        * Compute hazard h = sigmoid(logits)
        * Convert to PMF as in the cumulative link model
        * Sample categorical & shift back to the original label range
        """
        logits = self.forward(inputs)                     # (B,D,K)
        h     = self._hazard(logits)                      # (B,D,K)
        pmf   = self._pmf_from_hazard(h)                  # (B,D,V)

        samples = torch.distributions.Categorical(probs=pmf).sample()
        return samples + self.min_value                   # (B,D)
