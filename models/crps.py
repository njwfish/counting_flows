import torch
import torch.nn as nn
import torch.nn.functional as F

class CRPSLoss(nn.Module):
    """
    Discrete Flow model with CRPS loss.
    Works with arbitrary architectures that output (batch, data_dim, vocab_size) logits.
    """
    def __init__(self, architecture, min_value: int = 0, value_range: int = 127):
        """
        Args:
            architecture: nn.Module that on forward(**inputs) returns
                          logits of shape (B, D, V), where V = value_range+1.
            min_value: smallest integer in your support
            value_range: maximum integer minus min_value
        """
        super().__init__()
        self.arch = architecture
        self.min_value = min_value
        self.value_range = value_range
        print(f"[CRPSLoss] support = [{min_value} … {min_value+value_range}]")

    def forward(self, inputs):
        """
        Returns:
            logits (B, D, V)
        """
        return self.arch(**inputs)

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        Actually samples from the predicted distribution.
        """
        logits = self.forward(kwargs)  # (batch_size, data_dim, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        samples = torch.distributions.Categorical(probs=probs).sample()
        return samples + self.min_value 

    def loss(self, target, inputs):
        """
        Compute batch‐mean CRPS.

        Args:
            target: (B, D) integer targets in [min_value, min_value+value_range]
            inputs: passed to `architecture`

        Returns:
            scalar CRPS
        """
        # 1) get logits and convert to probabilities
        logits = self.forward(inputs)             # (B, D, V)
        probs  = F.softmax(logits, dim=-1)          # (B, D, V)

        # 2) cumulative distribution function
        cdf = probs.cumsum(dim=-1)                  # (B, D, V)

        # 3) clamp & shift targets to [0..V-1]
        target = target - self.min_value
        target = target.clamp(0, self.value_range).long()     # (B, D)

        # 4) build the “step” indicator 1{k ≥ target}
        #    we create a range [0…V-1] then compare against target
        V = self.value_range
        k = torch.arange(V, device=probs.device)    # (V,)
        # compare: broadcast (B,D,1) vs (V,) → (B,D,V)
        indicator = (k.unsqueeze(0).unsqueeze(0) >= target.unsqueeze(-1)).float()

        # 5) squared difference, sum over support, then mean over B,D
        crps = ((cdf - indicator) ** 2).sum(dim=-1) # (B, D)
        return crps.mean()                          # scalar
