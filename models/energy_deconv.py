import torch
import torch.nn as nn

import torch
import torch.nn.functional as F




@torch.no_grad()
def rescale(
    x_pred: torch.Tensor,           # [B, D] >=0 prior (e.g., softplus(logits))
    C: torch.Tensor,                # [G, D] target aggregates across G
    agg: torch.Tensor,              # [G, B] aggregation matrix (sparse or dense)
):
    """
    KL/I-projection y = argmin_{y>=0, sum_g y=C}
    
    Implements element-wise rescaling: x[i,d] * C[g,d] / sum_group(x[:,d])
    where g is the group that element i belongs to.

    Returns: y [B, D] nonnegative with exact column totals.
    """
    B, D = x_pred.shape
    G = C.shape[0]
    dev = x_pred.device
    dtype = x_pred.dtype

    # Compute group sums: [G, D]
    if agg.is_sparse:
        group_sums = torch.sparse.mm(agg, x_pred)  # [G, D]
    else:
        group_sums = agg @ x_pred  # [G, D]
    
    # Handle zero columns by setting them to 1 (avoid division by zero)
    group_sums = torch.where(group_sums == 0, torch.ones_like(group_sums), group_sums)
    
    # Compute scaling factors: C[g,d] / group_sums[g,d] for each group
    scale_factors = C / group_sums  # [G, D]
    
    # Map scaling factors back to individual elements
    # For each element x[i,d], we need scale_factors[g,d] where g is the group of element i
    if agg.is_sparse:
        # For sparse matrix, we need to map group indices back to element indices
        # agg is [G, B], we need the transpose operation to broadcast scaling
        agg_t = agg.t()  # [B, G] - each row has 1 in the column corresponding to its group
        
        # Broadcast scaling factors to all elements: [B, D]
        # agg_t @ scale_factors gives us the scale factor for each element
        element_scales = torch.sparse.mm(agg_t, scale_factors)  # [B, D]
    else:
        # For dense matrix, similar operation
        agg_t = agg.t()  # [B, G]
        element_scales = agg_t @ scale_factors  # [B, D]
    
    # Apply element-wise rescaling
    y = x_pred * element_scales  # [B, D]
    
    return y

@torch.no_grad()
def multinomial_counts_var_n(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Multinomial sampler with varying totals.
    Args:
      p: [BATCH, K] nonnegative; will be renormalized along dim=-1.
      n: [BATCH] nonnegative integers (total counts per row).
    Returns:
      counts: [BATCH, K] integer tensor; row-sums equal n.
    """
    eps = 1e-12
    B, K = p.shape
    p = p / (p.sum(dim=-1, keepdim=True).clamp_min(eps))

    n = n.to(torch.long)
    n_max = int(n.max().item())
    if n_max == 0:
        return torch.zeros(B, K, device=p.device, dtype=torch.long)

    # Draw n_max categorical samples per row (with replacement)
    idx = torch.multinomial(p, num_samples=n_max, replacement=True)  # [B, n_max]

    # Mask out positions beyond each row's n[b]
    pos = torch.arange(n_max, device=p.device).unsqueeze(0)          # [1, n_max]
    mask = (pos < n.unsqueeze(1)).to(p.dtype)                        # [B, n_max], 1.0 where kept

    # Scatter-add masked 1s into K bins -> exact integer counts
    counts = torch.zeros(B, K, device=p.device, dtype=p.dtype)
    counts = counts.scatter_add(1, idx, mask)                        # [B, K], float

    return counts.round().to(torch.long)


def multinomial_match_groups_torch(
    X: torch.Tensor,             # [B, D], nonnegative counts
    C: torch.Tensor,             # [G, D], target group totals
    group_id: torch.Tensor=None, # [B], int64 in [0..G-1]
    A: torch.Tensor=None,        # [G, B] 0/1 disjoint membership (optional)
    eps: float = 1e-12,
):
    """
    Samples Y [B, D] so that group-sums match C exactly, distributing each C[g,d]
    across members b in group g proportional to X[b,d] (uniform if that col is all-zero).
    No Python loop over K; uses one torch.multinomial + scatter_add per flattened (g,d) batch.
    """
    assert (group_id is not None) ^ (A is not None), "Provide exactly one of group_id or A."
    B, D = X.shape
    device = X.device
    if A is not None:
        group_id = A.argmax(dim=0)
        G = A.shape[0]
    else:
        G = C.shape[0]
    assert C.shape == (G, D)

    # Group-sort to make ragged groups contiguous
    order = torch.argsort(group_id)
    inv_order = torch.empty_like(order); inv_order[order] = torch.arange(B, device=device)
    gid_sorted = group_id[order]       # [B]
    X_sorted  = X[order]               # [B, D]
    sizes = torch.bincount(gid_sorted, minlength=G)  # [G]
    max_K = int(sizes.max().item())

    # Compute position within group (0..size_g-1) without loops
    offsets = torch.zeros(G+1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(sizes, dim=0)
    pos_within = torch.arange(B, device=device) - offsets[gid_sorted]

    # Pad to [G, max_K, D]
    X_pad = X_sorted.new_zeros((G, max_K, D))
    X_pad[gid_sorted, pos_within, :] = X_sorted

    # Build probs across members for each (g,d)
    denom = X_pad.sum(dim=1, keepdim=True)                      # [G, 1, D]
    P = torch.where(denom > 0, X_pad / (denom + eps), X_pad)    # temp; zero cols fixed below

    # Uniform fallback where the entire (g,d) is zero within the group
    zero_mask = (denom.squeeze(1) == 0)                         # [G, D]
    K_mask = (torch.arange(max_K, device=device)[None, :] < sizes[:, None])  # [G, max_K]
    size_f = sizes.clamp_min(1).to(P.dtype).view(G, 1, 1)
    uniform = K_mask.to(P.dtype)[:, None, :] / size_f           # [G, 1, max_K]
    P = P.permute(0, 2, 1)                                      # [G, D, max_K]
    P = torch.where(zero_mask[:, :, None], uniform.expand(G, D, max_K), P)

    # Flatten batch: [G*D, max_K]
    P_flat = P.reshape(-1, max_K)
    n_flat = C.reshape(-1).to(torch.long)

    # Sample counts per (g,d) across group members
    counts_flat = multinomial_counts_var_n(P_flat, n_flat)      # [G*D, max_K]

    # Reshape back to [G, max_K, D], mask padded members
    alloc = counts_flat.view(G, D, max_K).permute(0, 2, 1)      # [G, max_K, D]
    alloc = alloc * K_mask[:, :, None].to(alloc.dtype)

    # Unpad back to [B, D] in grouped order, then unsort to original order
    alloc_flat = alloc.reshape(G * max_K, D)
    flat_mask = K_mask.reshape(G * max_K)
    Y_grouped = alloc_flat[flat_mask]                           # [B, D]
    Y = torch.empty_like(X, dtype=alloc.dtype)
    Y[order] = Y_grouped

    # Ensure integer dtype
    return Y.to(torch.long)


@torch.no_grad()
def fully_vectorized_randomized_round_to_targets(
    x_float: torch.Tensor,           # [B, D] nonnegative floats
    C: torch.Tensor,                 # [G, D] target integer sums 
    agg: torch.Tensor,               # [G, B] aggregation matrix (sparse or dense)
) -> torch.Tensor:
    """
    Fully vectorized randomized rounding using multinomial_match_groups_torch.
    
    This achieves the ideal O(1) multinomial calls across all (group, dimension) pairs
    using the updated implementation that works around PyTorch's limitations.
    
    Args:
        x_float: [B, D] nonnegative float values to round
        C: [G, D] target integer sums for each group
        agg: [G, B] aggregation matrix mapping elements to groups
        
    Returns:
        y: [B, D] integer values with exact group sums matching C
    """
    # Step 1: Simple rounding to get initial integer values
    y_rounded = torch.round(x_float).long()
    
    # Step 2: Use multinomial_match_groups_torch to redistribute to exact targets
    # This redistributes proportional to current values and guarantees exact group sums
    
    if agg.is_sparse:
        # Convert sparse to dense for multinomial_match_groups_torch
        agg_dense = agg.to_dense()
    else:
        agg_dense = agg
    
    # multinomial_match_groups_torch redistributes counts to match exact targets
    y_exact = multinomial_match_groups_torch(
        X=y_rounded.float(),  # Current integer counts as starting distribution
        C=C.float(),          # Target group sums
        A=agg_dense           # Group membership matrix
    )
    
    return y_exact



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
        
        if 'A' in inputs:
            del inputs_with_noise['A']

        for key, value in inputs_with_noise.items():
            print(key, value.shape)
        
        return self.act_fn(self.architecture(**inputs_with_noise))

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
        predictions = self.act_fn(self.architecture(**replicated_inputs)).reshape(n, self.m, -1)
        
        # Apply sparse aggregation: [G, B] @ [B, m*D] → [G, m*D] → [G, m, D]
        B, m, d = predictions.shape
        predictions_flat = predictions.view(B, -1)  # [B, m*D] - flatten last two dims
        agg_predictions_flat = agg @ predictions_flat  # [G, m*D]
        predictions = agg_predictions_flat.view(agg.shape[0], m, d)  # [G, m, D]

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
        x_pred = self.act_fn(self.architecture(**flat_inputs))          # [B*G, D] (logits or reals)

        # Optional: normalize tiny columns to avoid degenerate zeros with positive targets
        # (ipf_robust_anchor also smooths)
        C = target_sum.to(device)

        # ----- 2) KL/I-projection (IPF) anchored to prior -----
        y_float = rescale(
            x_pred=x_pred,
            C=C,
            agg=agg
        ) 

        if return_float:
            return y_float

        # ----- 3) exact integerization across G per (B,D) -----
        y_int = fully_vectorized_randomized_round_to_targets(torch.round(y_float), C.to(torch.int64), agg)
        return y_int


