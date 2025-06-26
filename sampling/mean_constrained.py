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


import numpy as np
from scipy import special

def _hypergeom_support_np(k, N, K, n):
    """Boolean array – True where (k,N,K,n) lie in the hypergeometric support."""
    return (
        (k >= 0) & (k <= K) &          # successes drawn cannot exceed successes available
        (k <= n) &                     # …or number drawn
        (n <= N) & (K <= N) &          # parameter ordering
        (n - k <= N - K)               # failures drawn cannot exceed failures available
    )

def hypergeom_logpmf_numpy(k, N, K, n):
    """
    Numerically robust log-pmf of the hypergeometric distribution.

    – Broadcasts all four arguments to a common shape.
    – Returns –inf outside the support instead of NaN / +/-inf.
    """
    k, N, K, n = np.broadcast_arrays(k, N, K, n)        # shapes are now identical
    logp        = np.full(k.shape, -np.inf, dtype=np.float64)

    valid = _hypergeom_support_np(k, N, K, n)
    if np.any(valid):
        # Cast once to float64 for the gamma-log evaluations
        kv = k[valid].astype(np.float64)
        Nv = N[valid].astype(np.float64)
        Kv = K[valid].astype(np.float64)
        nv = n[valid].astype(np.float64)

        logp[valid] = (
            special.gammaln(Kv + 1)          - special.gammaln(kv + 1) - special.gammaln(Kv - kv + 1)
          + special.gammaln(Nv - Kv + 1)     - special.gammaln(nv - kv + 1)
          - special.gammaln(Nv - Kv - nv + kv + 1)
          - (special.gammaln(Nv + 1) - special.gammaln(nv + 1) - special.gammaln(Nv - nv + 1))
        )

    return logp

def _feasible_bounds(a, N_s, N_t, B_t):
    base_low   = a - N_s
    base_high  = a + N_s
    k_upper    = base_low + 2*B_t
    k_lower    = base_low + 2*np.maximum(0, N_s - N_t + B_t)

    lower = np.maximum(base_low,  k_lower)
    upper = np.minimum(base_high, k_upper)

    #  ❗️ real support: 0 ≤ k ≤ n and k ≤ K
    lower = np.clip(lower, 0, N_s)
    upper = np.clip(upper, 0, N_s)
    return lower, upper


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


import numpy as np
from scipy.special import gammaln

def _log_hypergeom(k, N, K, n):
    """
    Vectorised log-pmf for Hypergeom(N, K, n) evaluated at k.
    """
    return (
        gammaln(K + 1) - gammaln(k + 1) - gammaln(K - k + 1)
        + gammaln(N - K + 1) - gammaln(n - k + 1) - gammaln(N - K - n + k + 1)
        - gammaln(N + 1) + gammaln(n + 1) + gammaln(N - n + 1)
    )


def check_feasibility(B_s, N_t, B_t, N_s, S):
    """
    Returns a dict of boolean masks (and counts) for:
      • column‐sum mismatches
      • out‐of‐bounds entries
      • hypergeom‐support violations
    """
    B, d = B_s.shape

    # 1) Column‐sum mismatch
    col_sums   = B_s.sum(axis=0)
    sum_ok     = (col_sums == S)
    sum_diff   = col_sums - S

    # 2) Entry‐wise bounds
    lower = np.maximum(0, N_s + B_t - N_t)
    upper = np.minimum(N_s, B_t)
    too_small  = (B_s < lower)
    too_large  = (B_s > upper)

    # 3) Hypergeom support: B_s[i,j] must also satisfy
    #                   0 <= B_s <= B_t and 0 <= N_s - B_s <= N_t - B_t
    #    (equivalently:  max(0, N_s+B_t-N_t) <= B_s <= min(N_s, B_t) )
    hg_lower = lower
    hg_upper = upper
    hg_viol   = (B_s < hg_lower) | (B_s > hg_upper)

    # Bundle up
    report = {
      'sum_mismatch_mask': ~sum_ok,        # which columns
      'sum_diff':          sum_diff,       # by how much
      'too_small_mask':    too_small,      # entry‐wise
      'too_large_mask':    too_large,
      'hg_viol_mask':      hg_viol,        # entry‐wise
      'counts': {
        'n_cols_bad_sum':   int((~sum_ok).sum()),
        'n_entries_small':  int(too_small.sum()),
        'n_entries_large':  int(too_large.sum()),
        'n_entries_hg_viol':int(hg_viol.sum()),
      }
    }
    return report


def mh_mean_constrained_update_numpy(
    N_t: np.ndarray,     # (B,d) total population at time-t
    B_t: np.ndarray,     # (B,d) total successes at time-t
    N_s: np.ndarray,     # (B,d) sample size at time-s
    B_s: np.ndarray,     # (B,d) sample successes at time-s  ← **this** is both input & output
    S:   np.ndarray,     # (d,) target *mean* per column
    sweeps: int = 0,
    max_projections: int = 5,
    override_support: bool = False,
) -> np.ndarray:
    """
    MH update on B_s so that:
      • each entry stays in [0, min(N_s, B_t)],
      • each column‐sum = round(B * M[j]),
      • the hypergeometric target stays invariant.
    """
    B, d = B_s.shape
    # work in-place on a copy
    B_s = B_s.copy()
    lower = np.maximum(0, N_s + B_t - N_t)   # min successes possible
    upper = np.minimum(N_s, B_t)            # max successes possible

    # 0) quick projections to hit ∑_i B_s[i,j] = S_target[j]
    for _ in range(max_projections):
        B_s, _ = constrained_multinomial_proposal_numpy(
            N_t, B_t, N_s, B_s, S, override_support=False
        )
        # print("B_s", B_s.sum(axis=0), S)
        if np.all(B_s.sum(axis=0) == S):
            break
    # B_s, _ = constrained_multinomial_proposal_numpy(N_t, B_t, N_s, B_s, S, override_support=True)
    # print("B_s", B_s.sum(axis=0), S)

    # 1) MH sweeps via ±1 exchanges in each column
    rng  = np.random.default_rng()
    cols = np.arange(d)


    for _ in range(sweeps):
        # 1) feasibility masks
        can_inc = (B_s < upper)   # shape (B,d)
        can_dec = (B_s > lower)   # shape (B,d)

        # 2) how many rows we can inc/dec in each column
        cnt_inc = can_inc.sum(axis=0)  # (d,)
        cnt_dec = can_dec.sum(axis=0)  # (d,)

        # 3) which columns are even eligible?
        good = (cnt_inc > 0) & (cnt_dec > 0)  # (d,)
        if not good.any():
            continue

        cols = np.nonzero(good)[0]           # list of good j’s

        # 4) sample a random “rank” in each good column
        r_inc = rng.integers(0, cnt_inc[cols])   # how many trues to skip
        r_dec = rng.integers(0, cnt_dec[cols])

        # 5) cumsum trick: find the row where cumsum == rank+1
        #    restrict to only the good columns
        sub_inc = can_inc[:, cols]           # (B, n_good)
        sub_dec = can_dec[:, cols]

        csum_inc = np.cumsum(sub_inc, axis=0)  # (B, n_good)
        csum_dec = np.cumsum(sub_dec, axis=0)

        # pick p_rows and q_rows
        # we look for the first row where csum == r+1
        target_inc = r_inc[np.newaxis, :] + 1   # broadcast to (1,n_good)
        target_dec = r_dec[np.newaxis, :] + 1

        # this yields a boolean array (B, n_good), only one True per column
        pick_inc = (csum_inc == target_inc) & sub_inc
        pick_dec = (csum_dec == target_dec) & sub_dec

        # and argmax gives the row index
        p_rows = np.argmax(pick_inc, axis=0)  # shape (n_good,)
        q_rows = np.argmax(pick_dec, axis=0)

        # make sure p != q (if they collide, just skip that column)
        neq = (p_rows != q_rows)
        if not neq.any():
            continue

        sel     = np.nonzero(neq)[0]
        j_idx   = cols[sel]
        p_idx   = p_rows[sel]
        q_idx   = q_rows[sel]

        # 6) compute current/proposed counts
        curr_p = B_s[p_idx, j_idx]
        curr_q = B_s[q_idx, j_idx]
        prop_p = curr_p + 1
        prop_q = curr_q - 1

        # --- 6b) Re-filter to make sure prop_p, prop_q are in [lower,upper] ---
        p_lo = lower[p_idx, j_idx]
        p_hi = upper[p_idx, j_idx]
        q_lo = lower[q_idx, j_idx]
        q_hi = upper[q_idx, j_idx]

        support_ok = (
            (prop_p >= p_lo) & (prop_p <= p_hi) &
            (prop_q >= q_lo) & (prop_q <= q_hi)
        )
        if not support_ok.any():
            continue  # nothing valid this sweep

        # Select only the truly valid proposals
        sel        = support_ok.nonzero()[0]
        j_sel      = j_idx[sel]
        p_sel      = p_idx[sel]
        q_sel      = q_idx[sel]
        curr_p     = curr_p[sel]
        curr_q     = curr_q[sel]
        prop_p     = prop_p[sel]
        prop_q     = prop_q[sel]

        # --- 7) compute log-densities only on valid entries ---
        logp_curr = (
            hypergeom_logpmf_numpy(curr_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_numpy(curr_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )
        logp_prop = (
            hypergeom_logpmf_numpy(prop_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_numpy(prop_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )

        # --- 8) Clip and cap the MH ratio ---
        dlog  = logp_prop - logp_curr
        dlog  = np.clip(dlog, -50, +50)        # avoid overflow in exp
        ratio = np.exp(dlog)
        ratio = np.minimum(ratio, 1.0)         # α = min(1, ratio)


        accept = rng.random(len(ratio)) < ratio

        # print("numpy", "logp_curr", logp_curr.mean(),"logp_prop", logp_prop.mean(), "ratio", ratio.mean(), "accept", accept.mean())
        if not accept.any():
            continue

        # --- 9) Apply accepted moves ---
        a_sel = sel[accept]
        j_acc = j_idx[a_sel]
        p_acc = p_idx[a_sel]
        q_acc = q_idx[a_sel]

        B_s[p_acc, j_acc] += 1
        B_s[q_acc, j_acc] -= 1

    return B_s


@torch.no_grad()
def mh_mean_constrained_update(
    N_t:     torch.LongTensor,    # (B,d)
    B_t:     torch.LongTensor,    # (B,d)
    N_s:     torch.LongTensor,    # (B,d)
    B_s:     torch.LongTensor,    # (B,d)
    S:       torch.LongTensor,    # (d,)
    sweeps:  int = 10,
    backend: str = 'auto',
    override_support: bool = False
) -> torch.LongTensor:
    """
    Metropolis-Hastings mean constrained update with smart backend selection.
    
    Args:
        N_t, B_t, N_s, B_s, S: Input tensors/arrays
        sweeps: Number of sweeps
        backend: 'auto' (default), 'numpy', or 'torch'
        
    Returns:
        Same type as inputs: Updated x
    """
    # Determine input types
    inputs = [N_t, B_t, N_s, B_s, S]
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
        
        S_np = to_numpy(S).astype(np.int64)
        N_t_np = to_numpy(N_t).astype(np.int64)
        B_t_np = to_numpy(B_t).astype(np.int64)
        N_s_np = to_numpy(N_s).astype(np.int64)
        B_s_np = to_numpy(B_s).astype(np.int64)
        
        result_np = mh_mean_constrained_update_numpy(
            N_t_np, B_t_np, N_s_np, B_s_np, S_np, sweeps, override_support=override_support
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


import numpy as np
from typing import Tuple

def constrained_multinomial_proposal_numpy(
    N_t: np.ndarray,     # (B,d)
    B_t: np.ndarray,     # (B,d)
    N_s: np.ndarray,     # (B,d)
    B_s: np.ndarray,     # (B,d)
    S:   np.ndarray,     # (d,)
    override_support: bool = False,
) -> Tuple[np.ndarray,np.ndarray]:
    B, d = B_s.shape

    # 1) get true hyper‐geom support per‐entry
    lower = np.maximum(0,  N_s + B_t - N_t)   # (B,d)
    upper = np.minimum(N_s, B_t)              # (B,d)

    # 2) expected under hypergeom
    B_s_exp = np.where(N_t==0, 0.0, N_s * (B_t/N_t))

    # 3) column residuals & sign
    col_sum = B_s.sum(axis=0)     # (d,)
    R       = S - col_sum        # (d,)
    sgn     = np.sign(R).astype(int)  # (d,)
    Rabs    = np.abs(R).astype(int)   # (d,)

    # print("B_s_exp - S", np.sort(B_s_exp.sum(axis=0) - S))

    # 4) directional weights → probs
    #    we want to favor moves that shrink |B_s - E[B_s]|
    diff    = (B_s_exp - B_s) * sgn[np.newaxis,:]   # (B,d)
    w       = np.logaddexp(0, diff)                    # (B,d)
    probs   = w / w.sum(axis=0, keepdims=True)      # (B,d)

    # 5) vectorized multinomial for each column j
    rng     = np.random.default_rng()
    # transpose so each row is a p‐vector for one column
    p_mat   = probs.T       # shape (d,B)
    # pass array of counts Rabs (length d) and 2D p_mat → returns (d,B)
    draws   = rng.multinomial(Rabs, p_mat)
    # back to (B,d)
    delta   = draws.T       # (B,d)

    # 6) raw proposal and clip to true support
    B_prop = B_s + sgn[np.newaxis,:] * delta
    if not override_support:
        B_prop = np.clip(B_prop, lower, upper)

    return B_prop, sgn


def constrained_multinomial_proposal(
    N_t:     torch.LongTensor,    # (B,d)
    B_t:     torch.LongTensor,    # (B,d)
    N_s:     torch.LongTensor,    # (B,d)
    B_s:     torch.LongTensor,    # (B,d)
    S:       torch.LongTensor,    # (d,)
    backend: str = 'auto',
    override_support: bool = False
):
    """
    Constrained multinomial proposal with smart backend selection.
    
    Args:
        N_t, B_t, N_s, B_s, S: Input tensors/arrays
        backend: 'auto' (default), 'numpy', or 'torch'
        
    Returns:
        Same type as inputs: (x_prop, sgn)
    """
    # Determine input types
    inputs = [N_t, B_t, N_s, B_s, S]
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
        
        N_t_np = to_numpy(N_t).astype(np.int64)
        B_t_np = to_numpy(B_t).astype(np.int64)
        N_s_np = to_numpy(N_s).astype(np.int64)
        B_s_np = to_numpy(B_s).astype(np.int64)
        S_np = to_numpy(S).astype(np.int64)
        
        x_prop_np, sgn_np = constrained_multinomial_proposal_numpy(
            N_t_np, B_t_np, N_s_np, B_s_np, S_np, override_support=override_support
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
        
        N_t_torch = to_torch(N_t).long()
        B_t_torch = to_torch(B_t).long()
        N_s_torch = to_torch(N_s).long()
        B_s_torch = to_torch(B_s).long()
        S_torch = to_torch(S).long()
        
        # Move to same device
        device = N_t_torch.device
        for tensor in [B_t_torch, S_torch, N_s_torch, B_s_torch]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        N_t_torch = N_t_torch.to(device)
        B_t_torch = B_t_torch.to(device)
        S_torch = S_torch.to(device)
        N_s_torch = N_s_torch.to(device)
        B_s_torch = B_s_torch.to(device)
        
        B_prop_torch, sgn_torch = constrained_multinomial_proposal_torch(
            N_t_torch, B_t_torch, N_s_torch, B_s_torch, S_torch, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'numpy':
            B_prop = B_prop_torch.cpu().numpy()
            sgn = sgn_torch.cpu().numpy()
        else:
            B_prop = B_prop_torch
            sgn = sgn_torch
    
    return B_prop, sgn