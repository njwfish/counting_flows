import torch
import numpy as np
from scipy import special


def hypergeom_logpmf_torch(k, N, K, n):
    """Torch implementation of hypergeometric log PMF."""
    return (
        torch.lgamma(K+1) - torch.lgamma(k+1) - torch.lgamma(K-k+1)
      + torch.lgamma(N-K+1) - torch.lgamma(n-k+1)
      - torch.lgamma(N-K-n+k+1)
      - (torch.lgamma(N+1) - torch.lgamma(n+1) - torch.lgamma(N-n+1))
    )


def hypergeom_logpmf_numpy(k, N, K, n):
    """Numpy implementation of hypergeometric log PMF."""
    return (
        special.loggamma(K+1) - special.loggamma(k+1) - special.loggamma(K-k+1)
      + special.loggamma(N-K+1) - special.loggamma(n-k+1)
      - special.loggamma(N-K-n+k+1)
      - (special.loggamma(N+1) - special.loggamma(n+1) - special.loggamma(N-n+1))
    )


def hypergeom_logpmf(k, N, K, n, backend='auto'):
    """
    Hypergeometric log PMF with smart backend selection.
    
    Args:
        k, N, K, n: Parameters (tensors, arrays, or scalars)
        backend: 'auto' (default), 'numpy', or 'torch'
        
    Returns:
        Same type as inputs: log PMF values
    """
    # Determine input types
    inputs = [k, N, K, n]
    input_types = []
    devices = []
    
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            input_types.append('torch')
            devices.append(inp.device)
        elif isinstance(inp, np.ndarray):
            input_types.append('numpy')
            devices.append(None)
        else:
            input_types.append('scalar')
            devices.append(None)
    
    # Determine primary type and device
    if 'torch' in input_types:
        primary_type = 'torch'
        primary_device = None
        for device in devices:
            if device is not None:
                if primary_device is None or device.type == 'cuda':
                    primary_device = device
        if primary_device is None:
            primary_device = torch.device('cpu')
    elif 'numpy' in input_types:
        primary_type = 'numpy'
        primary_device = None
    else:
        primary_type = 'scalar'
        primary_device = None
    
    # Decide backend
    if backend == 'auto':
        use_backend = primary_type if primary_type in ['numpy', 'torch'] else 'torch'
    elif backend in ['numpy', 'torch']:
        use_backend = backend
    else:
        raise ValueError(f"backend must be 'auto', 'numpy', or 'torch', got {backend}")
    
    # Convert and dispatch
    if use_backend == 'numpy':
        # Convert to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.array(x)
        
        k_np = to_numpy(k)
        N_np = to_numpy(N)
        K_np = to_numpy(K)
        n_np = to_numpy(n)
        
        result_np = hypergeom_logpmf_numpy(k_np, N_np, K_np, n_np)
        
        # Convert back
        if primary_type == 'torch':
            result = torch.from_numpy(result_np).to(primary_device)
        elif primary_type == 'numpy':
            result = result_np
        else:  # scalar
            if result_np.ndim == 0:
                result = result_np.item()
            else:
                result = result_np
                
    else:  # torch backend
        # Convert to torch
        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k)
        elif not isinstance(k, torch.Tensor):
            k = torch.tensor(k)
            
        if isinstance(N, np.ndarray):
            N = torch.from_numpy(N)
        elif not isinstance(N, torch.Tensor):
            N = torch.tensor(N)
            
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        elif not isinstance(K, torch.Tensor):
            K = torch.tensor(K)
            
        if isinstance(n, np.ndarray):
            n = torch.from_numpy(n)
        elif not isinstance(n, torch.Tensor):
            n = torch.tensor(n)
        
        # Move to same device
        device = k.device
        for tensor in [N, K, n]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        k = k.to(device)
        N = N.to(device)
        K = K.to(device)
        n = n.to(device)
        
        result_torch = hypergeom_logpmf_torch(k, N, K, n)
        
        # Convert back
        if primary_type == 'numpy':
            result = result_torch.cpu().numpy()
        elif primary_type == 'torch':
            result = result_torch
        else:  # scalar
            if result_torch.numel() == 1:
                result = result_torch.item()
            else:
                result = result_torch.cpu().numpy()
    
    return result


def mh_mean_constrained_update_torch(
    x:       torch.LongTensor,    # (B,d)
    x_exp:   torch.Tensor,        # (B,d)
    S:       torch.LongTensor,    # (d,)
    a:       torch.LongTensor,    # (B,d)
    N_t:     torch.LongTensor,    # (B,d)
    B_t:     torch.LongTensor,    # (B,d)
    N_s:     torch.LongTensor,    # (B,d)
    sweeps:  int = 10,
    max_projections: int = 3,
    override_support: bool = False
) -> torch.LongTensor:
    """Torch implementation of MH mean constrained update."""
    B, d = x.shape
    device = x.device

    # Initial projection
    for _ in range(max_projections):
        x, _ = constrained_multinomial_proposal_torch(x, x_exp, S, a, N_s, override_support=override_support)
        if torch.all(x.sum(dim=0) == S):
            break

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
            hypergeom_logpmf_torch(Bsc_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            hypergeom_logpmf_torch(Bsc_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )
        logp_prop = (
            hypergeom_logpmf_torch(Bsp_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            hypergeom_logpmf_torch(Bsp_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )

        alpha = (logp_prop - logp_curr).exp().clamp(max=1.0)

        accept_mask = valid_mask & (torch.rand(d, device=device) < alpha)

        # Apply accepted proposals
        accept_idx = j_idx[accept_mask]
        x[p[accept_mask], accept_idx] += 1
        x[q[accept_mask], accept_idx] -= 1

    return x


def mh_mean_constrained_update_numpy(
    x:       np.ndarray,     # (B,d)
    x_exp:   np.ndarray,     # (B,d)
    S:       np.ndarray,     # (d,)
    a:       np.ndarray,     # (B,d)
    N_t:     np.ndarray,     # (B,d)
    B_t:     np.ndarray,     # (B,d)
    N_s:     np.ndarray,     # (B,d)
    sweeps:  int = 10,
    max_projections: int = 3,
    override_support: bool = False
) -> np.ndarray:
    """Numpy implementation of MH mean constrained update."""
    B, d = x.shape

    # Initial projection
    for _ in range(max_projections):
        x, _ = constrained_multinomial_proposal_numpy(x, x_exp, S, a, N_s, override_support=override_support)
        if np.all(x.sum(dim=0) == S):
            break

    for _ in range(sweeps):
        # Sample pairs of rows (p_j, q_j) for each column j
        p = np.random.randint(B, size=(d,))
        q = np.random.randint(B, size=(d,))
        neq_mask = (p != q)

        j_idx = np.arange(d)

        # Construct proposed x
        x_prop = x.copy()
        x_prop[p, j_idx] += 1
        x_prop[q, j_idx] -= 1

        # Check for non-negative entries
        support_mask = x_prop[q, j_idx] >= 0

        # Check |x - a| <= N_s constraint
        delta_p = np.abs(x_prop[p, j_idx] - a[p, j_idx])
        delta_q = np.abs(x_prop[q, j_idx] - a[q, j_idx])
        bound_mask = (delta_p <= N_s[p, j_idx]) & (delta_q <= N_s[q, j_idx])

        valid_mask = support_mask & bound_mask & neq_mask

        if not valid_mask.any():
            continue

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
            hypergeom_logpmf_numpy(Bsc_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            hypergeom_logpmf_numpy(Bsc_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )
        logp_prop = (
            hypergeom_logpmf_numpy(Bsp_p, N_t[p, j_idx], B_t[p, j_idx], N_s[p, j_idx]) +
            hypergeom_logpmf_numpy(Bsp_q, N_t[q, j_idx], B_t[q, j_idx], N_s[q, j_idx])
        )

        alpha = np.clip(np.exp(logp_prop - logp_curr), 0.0, 1.0)

        accept_mask = valid_mask & (np.random.rand(d) < alpha)

        # Apply accepted proposals
        accept_idx = j_idx[accept_mask]
        x[p[accept_mask], accept_idx] += 1
        x[q[accept_mask], accept_idx] -= 1

    return x


@torch.no_grad()
def mh_mean_constrained_update(
    x:       torch.LongTensor,    # (B,d)
    x_exp:   torch.Tensor,        # (B,d)
    S:       torch.LongTensor,    # (d,)
    a:       torch.LongTensor,    # (B,d)
    N_t:     torch.LongTensor,    # (B,d)
    B_t:     torch.LongTensor,    # (B,d)
    N_s:     torch.LongTensor,    # (B,d)
    sweeps:  int = 10,
    backend: str = 'auto',
    override_support: bool = False
) -> torch.LongTensor:
    """
    Metropolis-Hastings mean constrained update with smart backend selection.
    
    Args:
        x, x_exp, S, a, N_t, B_t, N_s: Input tensors/arrays
        sweeps: Number of sweeps
        backend: 'auto' (default), 'numpy', or 'torch'
        
    Returns:
        Same type as inputs: Updated x
    """
    # Determine input types
    inputs = [x, x_exp, S, a, N_t, B_t, N_s]
    input_types = []
    devices = []
    
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            input_types.append('torch')
            devices.append(inp.device)
        elif isinstance(inp, np.ndarray):
            input_types.append('numpy')
            devices.append(None)
        else:
            input_types.append('scalar')
            devices.append(None)
    
    # Determine primary type and device
    if 'torch' in input_types:
        primary_type = 'torch'
        primary_device = None
        for device in devices:
            if device is not None:
                if primary_device is None or device.type == 'cuda':
                    primary_device = device
        if primary_device is None:
            primary_device = torch.device('cpu')
    elif 'numpy' in input_types:
        primary_type = 'numpy'
        primary_device = None
    else:
        primary_type = 'scalar'
        primary_device = None
    
    # Decide backend
    if backend == 'auto':
        use_backend = primary_type if primary_type in ['numpy', 'torch'] else 'torch'
    elif backend in ['numpy', 'torch']:
        use_backend = backend
    else:
        raise ValueError(f"backend must be 'auto', 'numpy', or 'torch', got {backend}")
    
    # Convert and dispatch
    if use_backend == 'numpy':
        # Convert to numpy
        def to_numpy(inp):
            if isinstance(inp, torch.Tensor):
                return inp.cpu().numpy()
            elif isinstance(inp, np.ndarray):
                return inp
            else:
                return np.array(inp)
        
        x_np = to_numpy(x).astype(np.int64)
        x_exp_np = to_numpy(x_exp).astype(np.float64)
        S_np = to_numpy(S).astype(np.int64)
        a_np = to_numpy(a).astype(np.int64)
        N_t_np = to_numpy(N_t).astype(np.int64)
        B_t_np = to_numpy(B_t).astype(np.int64)
        N_s_np = to_numpy(N_s).astype(np.int64)
        
        result_np = mh_mean_constrained_update_numpy(
            x_np, x_exp_np, S_np, a_np, N_t_np, B_t_np, N_s_np, sweeps, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'torch':
            result = torch.from_numpy(result_np).to(primary_device).long()
        else:
            result = result_np
            
    else:  # torch backend
        # Convert to torch
        def to_torch(inp):
            if isinstance(inp, np.ndarray):
                return torch.from_numpy(inp)
            elif isinstance(inp, torch.Tensor):
                return inp
            else:
                return torch.tensor(inp)
        
        x_torch = to_torch(x).long()
        x_exp_torch = to_torch(x_exp).float()
        S_torch = to_torch(S).long()
        a_torch = to_torch(a).long()
        N_t_torch = to_torch(N_t).long()
        B_t_torch = to_torch(B_t).long()
        N_s_torch = to_torch(N_s).long()
        
        # Move to same device
        device = x_torch.device
        for tensor in [x_exp_torch, S_torch, a_torch, N_t_torch, B_t_torch, N_s_torch]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        x_torch = x_torch.to(device)
        x_exp_torch = x_exp_torch.to(device)
        S_torch = S_torch.to(device)
        a_torch = a_torch.to(device)
        N_t_torch = N_t_torch.to(device)
        B_t_torch = B_t_torch.to(device)
        N_s_torch = N_s_torch.to(device)
        
        result_torch = mh_mean_constrained_update_torch(
            x_torch, x_exp_torch, S_torch, a_torch, N_t_torch, B_t_torch, N_s_torch, sweeps, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'numpy':
            result = result_torch.cpu().numpy()
        else:
            result = result_torch
    
    return result


@torch.compile
def constrained_multinomial_proposal_torch(
    x:     torch.LongTensor,    # (B,d)
    x_exp: torch.Tensor,        # (B,d)
    S:     torch.LongTensor,    # (d,) target *column*-sums
    a:     torch.LongTensor,    # (B,d) current a = x0_hat_t
    N_s:   torch.LongTensor,    # (B,d) latent total jumps at s
    override_support: bool = False
):
    """Torch implementation of constrained multinomial proposal."""
    B, d = x.shape
    device = x.device
    
    # 1) compute column residuals
    col_sum = x.sum(dim=0)            # (d,)
    R       = S - col_sum            # (d,)
    sgn     = R.sign()         # (d,)
    Rabs    = R.abs()          # (d,)

    # 2) directional weights
    diff    = (x_exp - x).float() * sgn.unsqueeze(0)   # (B,d)
    # print("S", S)
    # print("col_sum", col_sum)
    # print("R", R)
    # print("diff", diff.max(), diff.min())
    # print("x_exp", x_exp, x_exp.max(), x_exp.min())
    # print("x", x, x.max(), x.min())
    weights = torch.exp(diff / 10) # diff.clamp(min=0.0)                     # zero out opp. dir
    wsum    = weights.sum(dim=0, keepdim=True)        # (1,d)
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
    if not override_support:
        lower = a - N_s
        upper = a + N_s
        x_prop = torch.max(torch.min(x_prop, upper), lower)

    return x_prop, sgn


def constrained_multinomial_proposal_numpy(
    x:     np.ndarray,    # (B,d)
    x_exp: np.ndarray,    # (B,d)
    S:     np.ndarray,    # (d,) target *column*-sums
    a:     np.ndarray,    # (B,d) current a = x0_hat_t
    N_s:   np.ndarray,    # (B,d) latent total jumps at s
    override_support: bool = False
):
    """Numpy implementation of constrained multinomial proposal."""
    B, d = x.shape
    
    # 1) compute column residuals
    col_sum = x.sum(axis=0)           # (d,)
    R       = S - col_sum            # (d,)
    sgn     = np.sign(R).astype(np.int64)  # (d,)
    Rabs    = np.abs(R).astype(np.int64)   # (d,)

    # 2) directional weights
    diff    = (x_exp - x).astype(np.float64) * sgn[np.newaxis, :]   # (B,d)
    weights = np.clip(diff, 0.0, None)                     # zero out opp. dir
    wsum    = weights.sum(axis=0, keepdims=True)           # (1,d)
    # avoid all-zero
    weights[:, wsum[0]==0] += 1.0
    wsum    = weights.sum(axis=0, keepdims=True)           # updated
    probs   = weights / wsum                               # (B,d)

    # 3) Multinomial sampling (vectorized across columns)
    delta = np.zeros_like(x)
    
    max_count = Rabs.max()
    if max_count > 0:
        for j in range(d):
            if Rabs[j] > 0:
                # Sample for this column
                samples = np.random.multinomial(Rabs[j], probs[:, j])
                delta[:, j] = samples

    # 4) raw proposal
    x_prop = x + sgn[np.newaxis, :] * delta               # (B,d)

    # 5) clip each entry to remain in support
    if not override_support:
        lower = a - N_s
        upper = a + N_s
        x_prop = np.maximum(np.minimum(x_prop, upper), lower)

    return x_prop, sgn


def constrained_multinomial_proposal(
    x:     torch.LongTensor,    # (B,d)
    x_exp: torch.Tensor,        # (B,d)
    S:     torch.LongTensor,    # (d,) target *column*-sums
    a:     torch.LongTensor,    # (B,d) current a = x0_hat_t
    N_s:   torch.LongTensor,    # (B,d) latent total jumps at s
    backend: str = 'auto',
    override_support: bool = False
):
    """
    Constrained multinomial proposal with smart backend selection.
    
    Args:
        x, x_exp, S, a, N_s: Input tensors/arrays
        backend: 'auto' (default), 'numpy', or 'torch'
        
    Returns:
        Same type as inputs: (x_prop, sgn)
    """
    # Determine input types
    inputs = [x, x_exp, S, a, N_s]
    input_types = []
    devices = []
    
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            input_types.append('torch')
            devices.append(inp.device)
        elif isinstance(inp, np.ndarray):
            input_types.append('numpy')
            devices.append(None)
        else:
            input_types.append('scalar')
            devices.append(None)
    
    # Determine primary type and device
    if 'torch' in input_types:
        primary_type = 'torch'
        primary_device = None
        for device in devices:
            if device is not None:
                if primary_device is None or device.type == 'cuda':
                    primary_device = device
        if primary_device is None:
            primary_device = torch.device('cpu')
    elif 'numpy' in input_types:
        primary_type = 'numpy'
        primary_device = None
    else:
        primary_type = 'scalar'
        primary_device = None
    
    # Decide backend
    if backend == 'auto':
        use_backend = primary_type if primary_type in ['numpy', 'torch'] else 'torch'
    elif backend in ['numpy', 'torch']:
        use_backend = backend
    else:
        raise ValueError(f"backend must be 'auto', 'numpy', or 'torch', got {backend}")
    
    # Convert and dispatch
    if use_backend == 'numpy':
        # Convert to numpy
        def to_numpy(inp):
            if isinstance(inp, torch.Tensor):
                return inp.cpu().numpy()
            elif isinstance(inp, np.ndarray):
                return inp
            else:
                return np.array(inp)
        
        x_np = to_numpy(x).astype(np.int64)
        x_exp_np = to_numpy(x_exp).astype(np.float64)
        S_np = to_numpy(S).astype(np.int64)
        a_np = to_numpy(a).astype(np.int64)
        N_s_np = to_numpy(N_s).astype(np.int64)
        
        x_prop_np, sgn_np = constrained_multinomial_proposal_numpy(
            x_np, x_exp_np, S_np, a_np, N_s_np, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'torch':
            x_prop = torch.from_numpy(x_prop_np).to(primary_device).long()
            sgn = torch.from_numpy(sgn_np).to(primary_device).long()
        else:
            x_prop = x_prop_np
            sgn = sgn_np
            
    else:  # torch backend
        # Convert to torch
        def to_torch(inp):
            if isinstance(inp, np.ndarray):
                return torch.from_numpy(inp)
            elif isinstance(inp, torch.Tensor):
                return inp
            else:
                return torch.tensor(inp)
        
        x_torch = to_torch(x).long()
        x_exp_torch = to_torch(x_exp).float()
        S_torch = to_torch(S).long()
        a_torch = to_torch(a).long()
        N_s_torch = to_torch(N_s).long()
        
        # Move to same device
        device = x_torch.device
        for tensor in [x_exp_torch, S_torch, a_torch, N_s_torch]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        x_torch = x_torch.to(device)
        x_exp_torch = x_exp_torch.to(device)
        S_torch = S_torch.to(device)
        a_torch = a_torch.to(device)
        N_s_torch = N_s_torch.to(device)
        
        x_prop_torch, sgn_torch = constrained_multinomial_proposal_torch(
            x_torch, x_exp_torch, S_torch, a_torch, N_s_torch, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'numpy':
            x_prop = x_prop_torch.cpu().numpy()
            sgn = sgn_torch.cpu().numpy()
        else:
            x_prop = x_prop_torch
            sgn = sgn_torch
    
    return x_prop, sgn