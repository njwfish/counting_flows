import torch


def hypergeom_logpmf(k,N,K,n):
    return (
        torch.lgamma(K+1) - torch.lgamma(k+1) - torch.lgamma(K-k+1)
      + torch.lgamma(N-K+1) - torch.lgamma(n-k+1)
      - torch.lgamma(N-K-n+k+1)
      - (torch.lgamma(N+1) - torch.lgamma(n+1) - torch.lgamma(N-n+1))
    )


@torch.no_grad()
def mh_mean_constrained_update(
    x:       torch.LongTensor,    # (B,d)
    x_exp:   torch.Tensor,        # (B,d)
    S:       torch.LongTensor,    # (d,)
    a:       torch.LongTensor,    # (B,d)
    N_t:     torch.LongTensor,    # (B,d)
    B_t:     torch.LongTensor,    # (B,d)
    N_s:     torch.LongTensor,    # (B,d)
    sweeps:  int = 10
) -> torch.LongTensor:
    B, d = x.shape
    device = x.device

    # Initial projection
    x, _ = constrained_multinomial_proposal(x, x_exp, S, a, N_s)

    for _ in range(sweeps):
        # Sample pairs of rows (p_j, q_j) for each column j
        p = torch.randint(B, (d,), device=device)
        q = torch.randint(B, (d,), device=device)
        neq_mask = (p != q)

        j_idx = torch.arange(d, device=device)

        # Construct proposed x
        x_prop = x.clone()
        x_prop[p, j_idx] += 1
        x_prop[q, j_idx] -= 1

        # Check for non-negative entries
        support_mask = x_prop[q, j_idx] >= 0

        # Check |x - a| <= N_s constraint
        delta_p = (x_prop[p, j_idx] - a[p, j_idx]).abs()
        delta_q = (x_prop[q, j_idx] - a[q, j_idx]).abs()
        bound_mask = (delta_p <= N_s[p, j_idx]) & (delta_q <= N_s[q, j_idx])

        valid_mask = support_mask & bound_mask & neq_mask

        if not valid_mask.any():
            continue

        # Compute hypergeometric log-probs (batched)
        def HG_log(k, N, K, n):
            return hypergeom_logpmf(k=k, N=N, K=K, n=n)

        # current and proposed birth counts
        def births(x_, a_, N_s_):
            return (N_s_ + (x_ - a_)) // 2

        x_pj  = x[p, j_idx]
        x_qj  = x[q, j_idx]
        xp_pj = x_prop[p, j_idx]
        xp_qj = x_prop[q, j_idx]

        # Current births
        Bsc_p = births(x_pj, a[p, j_idx], N_s[p, j_idx])
        Bsc_q = births(x_qj, a[q, j_idx], N_s[q, j_idx])
        # Proposed births
        Bsp_p = births(xp_pj, a[p, j_idx], N_s[p, j_idx])
        Bsp_q = births(xp_qj, a[q, j_idx], N_s[q, j_idx])

        logp_curr = (
            HG_log(Bsc_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            HG_log(Bsc_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )
        logp_prop = (
            HG_log(Bsp_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            HG_log(Bsp_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )

        alpha = (logp_prop - logp_curr).exp().clamp(max=1.0)

        accept_mask = valid_mask & (torch.rand(d, device=device) < alpha)

        # Apply accepted proposals
        accept_idx = j_idx[accept_mask]
        x[p[accept_mask], accept_idx] += 1
        x[q[accept_mask], accept_idx] -= 1

    return x


@torch.compile
def constrained_multinomial_proposal(
    x:     torch.LongTensor,    # (B,d)
    x_exp: torch.Tensor,        # (B,d)
    S:     torch.LongTensor,    # (d,) target *column*-sums
    a:     torch.LongTensor,    # (B,d) current a = x0_hat_t
    N_s:   torch.LongTensor,    # (B,d) latent total jumps at s
):
    """
    Scatter-add version inspired by torch.distributions.Multinomial.
    """
    B, d = x.shape
    device = x.device
    
    # 1) compute column residuals
    col_sum = x.sum(dim=0)            # (d,)
    R       = S - col_sum            # (d,)
    sgn     = R.sign().long()         # (d,)
    Rabs    = R.abs().long()          # (d,)

    # 2) directional weights
    diff    = (x_exp - x).float() * sgn.unsqueeze(0)   # (B,d)
    weights = diff.clamp(min=0.0)                     # zero out opp. dir
    wsum    = weights.sum(dim=0, keepdim=True)        # (1,d)
    # avoid all-zero
    weights[:, wsum[0]==0] += 1.0
    wsum    = weights.sum(dim=0, keepdim=True)        # updated
    probs   = weights / wsum                          # (B,d)

    # 3) Scatter-add multinomial sampling
    delta = torch.zeros_like(x)
    
    max_count = Rabs.max().item()
    if max_count > 0:
        # Step 1: Sample max_count times for ALL columns
        probs_T = probs.T  # (d, B)
        
        # Sample indices - this gives us which row (batch element) each sample goes to
        sample_indices = torch.multinomial(
            probs_T, 
            num_samples=max_count, 
            replacement=True
        )  # (d, max_count)
        
        # Step 2: Create mask for valid samples per column
        count_range = torch.arange(max_count, device=device).unsqueeze(0)  # (1, max_count)
        valid_mask = count_range < Rabs.unsqueeze(1)  # (d, max_count)
        
        # Step 3: Use scatter_add like torch.distributions.Multinomial
        # We need to scatter into a (B, d) tensor
        # Flatten the valid samples and their target locations
        
        # Get valid sample indices and their corresponding column indices
        valid_samples = sample_indices[valid_mask]  # (num_valid_samples,)
        
        # Get column indices for each valid sample
        col_indices = torch.arange(d, device=device).unsqueeze(1).expand(-1, max_count)
        valid_col_indices = col_indices[valid_mask]  # (num_valid_samples,)
        
        # Create flat indices into the (B, d) delta tensor
        # flat_idx = row_idx * d + col_idx
        flat_indices = valid_samples * d + valid_col_indices  # (num_valid_samples,)
        
        # Create flat delta tensor and scatter_add
        flat_delta = torch.zeros(B * d, dtype=torch.long, device=device)
        flat_delta.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
        
        # Reshape back to (B, d)
        delta = flat_delta.view(B, d)

    # 4) raw proposal
    x_prop = x + sgn.unsqueeze(0) * delta             # (B,d)

    # 5) clip each entry to remain in support
    lower = a - N_s
    upper = a + N_s
    x_prop = torch.max(torch.min(x_prop, upper), lower)

    return x_prop, sgn