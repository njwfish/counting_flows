import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

from .energy import EnergyScoreLoss

@torch.no_grad()
def rescale(
    x: torch.Tensor,        # [B, D] >= 0
    C: torch.Tensor,        # [G, D] >= 0
    A: torch.Tensor,        # [G, B] 0/1, exactly one 1 per column
) -> torch.Tensor:
    """
    If (A @ x)[g,d] > 0:  y[b,d] = x[b,d] * C[g,d] / (A@x)[g,d]  for b in group g
    If (A @ x)[g,d] = 0:  y[b,d] = C[g,d] / |g|                   uniformly for b in g
    Guarantees (A @ y) == C (up to fp), y >= 0. Fully vectorized.
    """
    # Dense view for argmax/gather; sparse (coalesced) is fine for matmul
    if getattr(A, "is_sparse", False):
        A_sp = A.coalesce().to(torch.float32)
        A_dense = A_sp.to_dense()
    else:
        A_sp = A.to(torch.float32)
        A_dense = A.to(torch.float32)

    # Sanity checks for disjointness
    assert torch.all((A_dense == 0) | (A_dense == 1)), "A must be 0/1."
    assert torch.all(A_dense.sum(dim=0) == 1), "Each item must belong to exactly one group."

    B, D = x.shape
    G, B2 = A_dense.shape
    assert B2 == B and C.shape == (G, D)

    x64 = x.to(torch.float32)
    C64 = C.to(torch.float32)

    # Group ids and sizes
    gid = A_dense.argmax(dim=0)                       # [B]
    sizes = A_dense.sum(dim=1).to(torch.float32)      # [G]
    if torch.any((sizes == 0) & (C64.sum(dim=1) != 0)):
        raise ValueError("Group with zero members has positive target in C.")

    # Group sums per (g,d)
    Ax = (A_sp @ x64) if getattr(A, "is_sparse", False) else (A_dense @ x64)  # [G, D]

    # Positive / zero masks
    pos  = Ax > 0.5
    zero = ~pos

    # Scale factors for positive (g,d)
    scale = torch.zeros_like(C64)
    scale[pos] = C64[pos] / Ax[pos]                   # [G, D]

    # Gather each row's group scale (don’t sum across groups)
    y = (x64 * scale[gid])                            # [B, D]

    # Uniform fallback for zero groups: share[g,d] = C[g,d] / |g|
    if torch.any(zero):
        # Broadcast sizes to (G,D), divide everywhere, then mask to only zero cells
        safe_sizes = torch.where(sizes > 0, sizes, torch.ones_like(sizes))    # [G]
        share = (C64 / safe_sizes[:, None]) * zero.to(C64.dtype)              # [G, D]
        # Distribute to members via A^T
        y += (A_dense.t() @ share)                                            # [B, D]

    return y.to(x.dtype)

@torch.no_grad()
def randomized_round_groups_exact(
    X: torch.Tensor,         # [B, D] nonnegative floats, already scaled so group-sums match C
    C: torch.Tensor,         # [G, D] integer targets (exact group sums)
    A: torch.Tensor,         # [G, B] 0/1 disjoint membership matrix (one 1 per column)
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Integer rounding with exact group targets and ||Y - X||_inf <= 1.
    For each (g,d), choose S[g,d] = C[g,d] - sum_b floor(X[b,d]) members (without replacement)
    with probabilities proportional to frac(X[b,d]); add +1 to those entries.

    Assumes: A is 0/1 with columns summing to 1 (disjoint membership);
             C is integer & nonnegative; X is nonnegative and sums per (g,d) match C exactly.
    """

    device = X.device
    B, D = X.shape
    G, B2 = A.shape
    assert B2 == B, "A must be [G, B] matching X's batch size"
    assert torch.all((A == 0) | (A == 1)), "A must be 0/1"
    assert torch.all(A.sum(dim=0) == 1), "Each item must belong to exactly one group"
    assert torch.all(C >= 0)
    # (Optional) strict integer check for C:
    assert torch.allclose(C, C.round()), "C must be integer-valued"

    # Derive group ids and sizes; sort by group to make ragged groups contiguous
    group_id = A.argmax(dim=0)                          # [B]
    order = torch.argsort(group_id)                     # [B]
    inv_order = torch.empty_like(order); inv_order[order] = torch.arange(B, device=device)
    gid_sorted = group_id[order]                        # [B]
    X_sorted  = X[order]                                # [B, D]
    sizes = torch.bincount(gid_sorted, minlength=G)     # [G]
    max_K = int(sizes.max().item())

    # Guard: empty group cannot have positive target
    if torch.any((sizes == 0) & (C.sum(dim=1) > 0)):
        raise ValueError("Empty group with positive targets in C.")

    # Compute position within group (0..size_g-1)
    offsets = torch.zeros(G+1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(sizes, dim=0)
    pos_within = torch.arange(B, device=device) - offsets[gid_sorted]

    # Pad to [G, max_K, D]
    X_pad = X_sorted.new_zeros((G, max_K, D))
    X_pad[gid_sorted, pos_within, :] = X_sorted
    K_mask = (torch.arange(max_K, device=device)[None, :] < sizes[:, None])   # [G, max_K]

    # Integer/fractional parts
    X_floor = torch.floor(X_pad)                           # [G, max_K, D]
    frac    = (X_pad - X_floor).clamp_min(0.0)            # [G, max_K, D]
    frac = frac * K_mask[:, :, None]                      # zero out padding

    # Required number of +1 per (g,d): S = C - sum floor(X)
    floor_sums = X_floor.sum(dim=1)                       # [G, D]
    S = (C.to(X.dtype) - floor_sums).round().long()  # [G, D], should be integral already
    if torch.any(S < 0):
        raise ValueError("Negative S encountered; X not consistent with C?")
    if torch.any(S > sizes[:, None]):
        raise ValueError("S exceeds group size; check X/C consistency.")

    # If S[g,d] == 0, nothing to add; if all frac==0 then S must be 0 by consistency.

    # Weighted sampling WITHOUT replacement via Gumbel-Top-k, vectorized across all (g,d).
    # scores = log(frac) + Gumbel(0,1); take top S[g,d] indices.
    tiny = eps
    # logits: -inf where frac==0 or padding
    logits = torch.where(frac > 0, torch.log(frac + tiny), frac.new_full((), float('-inf')))
    # Gumbel noise
    U = torch.rand_like(logits, dtype=logits.dtype)
    gumbel = -torch.log(-torch.log(U.clamp_min(tiny)))
    scores = logits + gumbel

    # Get descending order along the member axis
    # ord: [G, max_K, D] giving permutation of member indices for each (g,d)
    ord = torch.argsort(scores, dim=1, descending=True)

    # Compute rank (0 is best) for each member
    ranks = torch.empty_like(ord, dtype=torch.long)
    arange_idx = torch.arange(max_K, device=device).view(1, max_K, 1).expand(G, max_K, D)
    ranks.scatter_(1, ord, arange_idx)  # ranks[g, member, d] = position in sorted order

    # Select top S[g,d] members: mask where rank < S[g,d]
    select_mask = ranks < S[:, None, :]                  # [G, max_K, D], bool
    select_mask = select_mask & K_mask[:, :, None]       # keep only real members

    # Build Y: floor + 1 on selected entries
    Y_pad = X_floor + select_mask.to(X_floor.dtype)

    # Unpad back to [B, D] (grouped order), then unsort
    Y_grouped = Y_pad[K_mask, :].view(B, D)
    Y = torch.empty_like(X_sorted)                       # [B, D]
    Y = Y_grouped
    Y_out = torch.empty_like(X)
    Y_out[order] = Y
    return Y_out.to(torch.long)


class DeconvolutionEnergyScoreLoss(EnergyScoreLoss):
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
    ):
        super().__init__(
            architecture=architecture,
            noise_dim=noise_dim,
            m_samples=m_samples,
            lambda_energy=lambda_energy,
        )

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
        
        if 'A' in inputs:
            del inputs_with_noise['A']

        for key, value in inputs_with_noise.items():
            print(key, value.shape)
        
        return self.architecture(**inputs_with_noise)

    def loss(self, target, inputs, agg):
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

        # Replicate all inputs m times by iterating over the dict
        if 'A' in inputs:
            del inputs['A']
        replicated_inputs = {}
        for key, value in inputs.items():
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.architecture(**replicated_inputs).reshape(n, self.m, -1)
        
        # Apply sparse aggregation: [G, B] @ [B, m*D] → [G, m*D] → [G, m, D]
        B, m, d = predictions.shape
        predictions_flat = predictions.view(B, -1)  # [B, m*D] - flatten last two dims
        agg_predictions_flat = agg @ predictions_flat  # [G, m*D]
        predictions = agg_predictions_flat.view(agg.shape[0], m, d)  # [G, m, D]

        return self._compute_energy_score(predictions, target, λ)


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
        agg = inputs['A']
        device = base.device
        BG = base.shape[0]
        B, D = target_sum.shape
        E = self.noise_dim

        # ----- 1) sample prior x_pred from architecture with noise -----
        flat_inputs = {k: v for k, v in inputs.items()}
        noise = torch.randn(BG, E, device=device)
        flat_inputs['noise'] = noise
        if 'A' in flat_inputs:
            del flat_inputs['A']
        x_pred = self.architecture(**flat_inputs)          # [B*G, D] (logits or reals)

        # Optional: normalize tiny columns to avoid degenerate zeros with positive targets
        # (ipf_robust_anchor also smooths)
        C = target_sum.to(device)

        # ----- 2) KL/I-projection (IPF) anchored to prior -----
        y_float = rescale(
            x=x_pred,
            C=C,
            A=agg
        )

        # print(y_float.float().var(dim=0))

        if return_float:
            return y_float

        # ----- 3) exact integerization across G per (B,D) -----
        y_int = randomized_round_groups_exact(y_float, C, agg)
        # print(y_int.float().var(dim=0))
        return y_int


