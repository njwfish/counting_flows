import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

import math
@torch.no_grad()
def _round_along_G_to_targets_safe(x: torch.Tensor, target_bd: torch.Tensor) -> torch.Tensor:
    """
    x: [B,G,D] nonnegative floats
    target_bd: [B,D] integer totals across G
    Returns y: [B,G,D] ints with exact column sums.
    Robust to large residuals via quotient–remainder split.
    """
    B, G, D = x.shape
    dev = x.device

    x = x.clamp_min(0.0)
    xf = torch.floor(x)
    y = xf.to(torch.int64)
    frac = x - xf

    need = target_bd.to(torch.int64) - y.sum(dim=1)        # [B,D]

    # Uniform base addition if need > G
    pos = need.clamp_min(0)
    if pos.any():
        q = torch.div(pos, G, rounding_mode='floor')       # [B,D]
        if q.any():
            y = y + q.unsqueeze(1)
        r = pos - q * G                                    # [B,D], r < G
        k = int(r.max().item()) if r.numel() else 0
        if k > 0:
            k = min(k, G)
            idx = torch.topk((frac).permute(0,2,1), k=k, dim=-1, largest=True).indices  # [B,D,k]
            rowmask = (torch.arange(k, device=dev).view(1,1,-1) < r.unsqueeze(-1)).to(torch.int64)
            add = torch.zeros_like(y.permute(0,2,1))        # [B,D,G]
            add.scatter_add_(-1, idx, rowmask)
            y = (y.permute(0,2,1) + add).permute(0,2,1)

    # Recompute remaining need (should be <= 0 now)
    need = target_bd.to(torch.int64) - y.sum(dim=1)
    neg = (-need).clamp_min(0)
    if neg.any():
        # Bound by availability
        avail_tot = y.sum(dim=1)
        neg = torch.minimum(neg, avail_tot)

        # Uniform base removal
        q = torch.div(neg, G, rounding_mode='floor')
        if q.any():
            y_before = y
            y = torch.clamp(y - q.unsqueeze(1), min=0)
            removed = (y_before - y).sum(dim=1)
            neg = neg - removed

        # Final small remainders: remove from smallest frac where y>0
        if neg.any():
            r = neg
            k = int(r.max().item()) if r.numel() else 0
            if k > 0:
                k = min(k, G)
                big = torch.full_like(frac, float('inf'))
                frac_masked = torch.where(y > 0, frac, big)
                idx = torch.topk(frac_masked.permute(0,2,1), k=k, dim=-1, largest=False).indices
                rowmask = (torch.arange(k, device=dev).view(1,1,-1) < r.unsqueeze(-1)).to(torch.int64)
                rem = torch.zeros_like(y.permute(0,2,1))
                rem.scatter_add_(-1, idx, rowmask)
                y = (y.permute(0,2,1) - rem).clamp_min_(0).permute(0,2,1)

    # Optional debug check:
    # assert torch.all(y.sum(dim=1) == target_bd), "aggregate mismatch"
    return y


@torch.no_grad()
def rescale(
    x_pred: torch.Tensor,           # [B,G,D] >=0 prior (e.g., softplus(logits))
    C: torch.Tensor,                # [B,D] target aggregates across G
):
    """
    KL/I-projection y = argmin_{y>=0, sum_g y=C, (and sum_d y=R if keep_rows)}
    where R = row totals of x_pred (anchor). Preserves odds ratios (dependence).
    Multiplicative scaling ⇒ small deltas when C is close to current columns.

    Returns: y [B,G,D] nonnegative with exact column totals; rows exact if feasible.
    """
    B, G, D = x_pred.shape
    dev = x_pred.device
    dtype = x_pred.dtype

    group_sums = x_pred.sum(dim=1, keepdim=True) # shape [B, 1, D]
    x_pred[(group_sums == 0)] = 1

    # recompute group sums with no zero groups
    group_sums = x_pred.sum(dim=1, keepdim=True)
    scale_matrix = x_pred / group_sums # shape [B, G, D]
    y = C * scale_matrix
    return y


class DeconvolutionEnergyScoreLoss(nn.Module):
    """
    Distributional-diffusion energy score loss (eq.14) with m-sample approximation.
    Works with arbitrary architectures and clean input interface.
    """
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
        min_value: int = 0,
        value_range: int = 10,
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy
        self.min_value = min_value
        self.value_range = value_range
        if self.min_value == 0:
            self.act_fn = nn.Softplus()
        else:
            self.act_fn = nn.Identity()

    def _pairwise_dist(self, a, b, eps=1e-6):
        """
        Compute pairwise distances for energy score
        a: [n, d], b: [m, d] → [n, m] of √(||a_i - b_j||² + eps)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)      # [n, m, d]
        sq   = (diff * diff).sum(-1)                # [n, m]
        return torch.sqrt(torch.clamp(sq, min=eps))

    def forward(self, inputs, eps=None):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            
        Returns:
            Prediction tensor
        """
        # Add noise (energy score specific requirement)
        inputs_with_noise = inputs.copy()
        base_input = list(inputs.values())[0]
        batch_size = base_input.shape[0]
        if eps is None:
            inputs_with_noise['noise'] = torch.randn(batch_size, self.noise_dim, device=base_input.device)
        else:
            inputs_with_noise['noise'] = eps
        
        return self.act_fn(self.architecture(**inputs_with_noise))

    def loss(self, target, inputs):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||target_i - pred_ij||
            - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
          ]
        """
        base_input = list(inputs.values())[0]
        n, λ = base_input.shape[0], self.lambda_energy
        num_groups = target.shape[0]
        group_size = n // num_groups

        # Replicate all inputs m times by iterating over the dict
        replicated_inputs = {}
        for key, value in inputs.items():
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.act_fn(self.architecture(**replicated_inputs))
        predictions = predictions.view(num_groups, group_size, self.m, -1)  # [n, num_groups, m, d]
        predictions = predictions.sum(dim=1)  # [n, m, d]

        # Confinement term: distance to target
        target_expanded = target.unsqueeze(1).expand(-1, self.m, -1)
        term_conf = (predictions - target_expanded).norm(dim=2).mean(dim=1)  # [n]

        # Interaction term (efficient batched computation)
        # Using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ identity
        sq = predictions.pow(2).sum(dim=2)  # [n, m] - squared norms
        inn = torch.bmm(predictions, predictions.transpose(1,2))  # [n, m, m] - inner products
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2*inn  # [n, m, m] - squared distances
        sqd = torch.clamp(sqd, min=1e-6)  # avoid sqrt(0)
        d = sqd.sqrt()  # [n, m, m] - distances
        
        # Mean of off-diagonal pairwise distances
        # Create mask for off-diagonal elements on the fly
        m_mask = torch.ones(self.m, self.m, device=predictions.device) - torch.eye(self.m, device=predictions.device)
        mean_pd = (d * m_mask).sum(dim=(1,2)) / (self.m * (self.m - 1))  # [n]
        term_int = (λ / 2.0) * mean_pd  # [n]

        return (term_conf - term_int).mean()

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        """
        if 'target_sum' in kwargs:
            S = kwargs['target_sum']
            del kwargs['target_sum']
            return self.conditional_sample(kwargs, S)
        else:
            prediction = self.forward(kwargs)
            return prediction.round().long()

    @torch.no_grad()
    def conditional_sample(
        self,
        inputs: dict,              # tensors shaped [B*G, ...]; no 'noise' key required
        target_sum: torch.Tensor,  # [B, D] aggregates across G (per batch item)
        keep_rows: bool = True,    # anchor each individual's total mass
        trust_clip: float | None = 0.25,  # limit per-iter scaling to ±25% (stability)
        smooth_eps: float = 1e-8,
        ipf_iters: int = 60,
        tol: float = 1e-7,
        return_float: bool = False # if True: return non-integer y with exact columns
    ):
        """
        Allocation-based conditional sampler (no latent optimization):
        1) Sample prior counts x_pred from the model (with fresh noise).
        2) KL/I-projection (IPF) onto {sum_g y = target_sum} (and row totals if keep_rows=True).
        3) Exact integerization across G per (B,D).

        • Multiplicative/IPF updates ⇒ small deltas when targets are close (anchored to x_pred).
        • keep_rows=True preserves each individual's total mass from x_pred (strong anchoring).
        """

        # ----- shapes -----
        base = next(iter(inputs.values()))
        device = base.device
        BG = base.shape[0]
        B, D = target_sum.shape
        assert BG % B == 0, f"inputs first dim {BG} must be B*G; B={B}"
        G = BG // B
        E = self.noise_dim

        # ----- 1) sample prior x_pred from architecture with noise -----
        flat_inputs = {k: v for k, v in inputs.items()}
        noise = torch.randn(BG, E, device=device)
        flat_inputs['noise'] = noise
        x_pred = self.act_fn(self.architecture(**flat_inputs)).reshape(B, G, D)          # [B*G, D] (logits or reals)

        # Optional: normalize tiny columns to avoid degenerate zeros with positive targets
        # (ipf_robust_anchor also smooths)
        C = target_sum.to(device)

        # ----- 2) KL/I-projection (IPF) anchored to prior -----
        y_float = rescale(
            x_pred=x_pred,
            C=C,
        ) 

        if return_float:
            return y_float.reshape(B * G, D)  # exact aggregates, minimal KL change

        # ----- 3) exact integerization across G per (B,D) -----
        y_int = _round_along_G_to_targets_safe(y_float, C.to(torch.int64))
        return y_int.reshape(B * G, D)


