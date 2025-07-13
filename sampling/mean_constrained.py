import torch
torch._dynamo.config.capture_scalar_outputs = True

import numpy as np
from scipy import special
from typing import Union, Tuple, Optional
import time

import numpy as np
from scipy import special

from .multinomial import multinomial

from ..utils import maybe_compile

def hypergeom_logpmf_torch(k, N, K, n):
    """Torch implementation of hypergeometric log PMF."""

    return (
        torch.lgamma(K+1) - torch.lgamma(k+1) - torch.lgamma(K-k+1)
      + torch.lgamma(N-K+1) - torch.lgamma(n-k+1) 
      - torch.lgamma(N-K-n+k+1)
      - (torch.lgamma(N+1) - torch.lgamma(n+1) - torch.lgamma(N-n+1))
    )

def hypergeom_logpmf_numpy(k, N, K, n):
    """
    Numerically robust log-pmf of the hypergeometric distribution.

    – Broadcasts all four arguments to a common shape.
    – Returns –inf outside the support instead of NaN / +/-inf.
    """
    logp = (
        special.gammaln(K + 1)          - special.gammaln(k + 1) - special.gammaln(K - k + 1)
        + special.gammaln(N - K + 1)     - special.gammaln(n - k + 1)
        - special.gammaln(N - K - n + k + 1)
        - (special.gammaln(N + 1) - special.gammaln(n + 1) - special.gammaln(N - n + 1))
    )

    return logp


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

@torch.no_grad
@maybe_compile
def mh_mean_constrained_update_torch(
    N_t:     torch.LongTensor,    # (B,d) total population at time-t
    B_t:     torch.LongTensor,    # (B,d) total successes at time-t
    N_s:     torch.LongTensor,    # (B,d) sample size at time-s
    B_s:     torch.LongTensor,    # (B,d) sample successes at time-s
    S:       torch.LongTensor,    # (d,)  target sum per column
    sweeps:  int = 10,
    max_projections: int = 5,
    override_support: bool = False
) -> torch.LongTensor:
    """Torch implementation of MH mean constrained update."""
    B, d = B_s.shape
    device = B_s.device

    B_s = B_s.clone()
    lower = torch.maximum(torch.tensor(0, device=device), N_s + B_t - N_t)
    upper = torch.minimum(N_s, B_t)

    for _ in range(max_projections):
        B_s, _ = constrained_multinomial_proposal_torch(
            N_t, B_t, N_s, B_s, S, override_support=False
        )
        if torch.all(B_s.sum(dim=0) == S):
            break

    for _ in range(sweeps):
        can_inc = (B_s < upper)
        can_dec = (B_s > lower)

        cnt_inc = can_inc.sum(dim=0)
        cnt_dec = can_dec.sum(dim=0)

        good = (cnt_inc > 0) & (cnt_dec > 0)
        if not good.any():
            continue

        cols = torch.nonzero(good).squeeze(1)
        n_good = len(cols)

        r_inc = torch.randint(
            high=cnt_inc[cols].max().item(), size=(n_good,), device=device
        ) % cnt_inc[cols]
        r_dec = torch.randint(
            high=cnt_dec[cols].max().item(), size=(n_good,), device=device
        ) % cnt_dec[cols]

        sub_inc = can_inc[:, cols]
        sub_dec = can_dec[:, cols]

        csum_inc = torch.cumsum(sub_inc, dim=0)
        csum_dec = torch.cumsum(sub_dec, dim=0)

        target_inc = r_inc.unsqueeze(0) + 1
        target_dec = r_dec.unsqueeze(0) + 1

        pick_inc = (csum_inc == target_inc) & sub_inc
        pick_dec = (csum_dec == target_dec) & sub_dec

        p_rows = torch.argmax(pick_inc.long(), dim=0)
        q_rows = torch.argmax(pick_dec.long(), dim=0)
        
        neq = (p_rows != q_rows)
        if not neq.any():
            continue
        
        sel = torch.nonzero(neq).squeeze(1)
        j_idx = cols[sel]
        p_idx = p_rows[sel]
        q_idx = q_rows[sel]
        
        curr_p = B_s[p_idx, j_idx]
        curr_q = B_s[q_idx, j_idx]
        prop_p = curr_p + 1
        prop_q = curr_q - 1

        p_lo = lower[p_idx, j_idx]
        p_hi = upper[p_idx, j_idx]
        q_lo = lower[q_idx, j_idx]
        q_hi = upper[q_idx, j_idx]

        support_ok = (prop_p >= p_lo) & (prop_p <= p_hi) & (prop_q >= q_lo) & (prop_q <= q_hi)
        if not support_ok.any():
            continue

        sel_supp = torch.nonzero(support_ok).squeeze(1)
        j_sel = j_idx[sel_supp]
        p_sel = p_idx[sel_supp]
        q_sel = q_idx[sel_supp]
        curr_p = curr_p[sel_supp]
        curr_q = curr_q[sel_supp]
        prop_p = prop_p[sel_supp]
        prop_q = prop_q[sel_supp]

        logp_curr = (
            hypergeom_logpmf_torch(curr_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_torch(curr_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )
        logp_prop = (
            hypergeom_logpmf_torch(prop_p, N_t[p_sel,j_sel], B_t[p_sel,j_sel], N_s[p_sel,j_sel]) +
            hypergeom_logpmf_torch(prop_q, N_t[q_sel,j_sel], B_t[q_sel,j_sel], N_s[q_sel,j_sel])
        )

        dlog = logp_prop - logp_curr
        dlog = torch.clamp(dlog, -50, 50)
        ratio = torch.exp(dlog)
        ratio = torch.minimum(ratio, torch.tensor(1.0, device=device))

        accept = torch.rand(len(ratio), device=device) < ratio

        # print("torch", time.time() - time_start, "logp_curr", logp_curr.mean(),"logp_prop", logp_prop.mean(), "ratio", ratio.mean(), "accept", accept.float().mean())
        if not accept.any():
            continue

        a_sel = torch.nonzero(accept).squeeze(1)
        j_acc = j_sel[a_sel]
        p_acc = p_sel[a_sel]
        q_acc = q_sel[a_sel]
        
        # In-place updates can be tricky with indexing, use a safe way
        B_s.index_put_((p_acc, j_acc), B_s[p_acc, j_acc] + 1)
        B_s.index_put_((q_acc, j_acc), B_s[q_acc, j_acc] - 1)

    return B_s

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

    # 1) MH sweeps via ±1 exchanges in each column
    rng  = np.random.default_rng()

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

        cols = np.nonzero(good)[0]           # list of good j's

        # 4) sample a random "rank" in each good column
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

        print("numpy", time.time() - time_start, "logp_curr", logp_curr.mean(),"logp_prop", logp_prop.mean(), "ratio", ratio.mean(), "accept", accept.mean())
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


@torch.no_grad
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
        
        N_t_torch = to_torch(N_t).long()
        B_t_torch = to_torch(B_t).long()
        N_s_torch = to_torch(N_s).long()
        B_s_torch = to_torch(B_s).long()
        S_torch = to_torch(S).long()
        
        # Move to same device
        device = B_s_torch.device
        for tensor in [N_t_torch, B_t_torch, N_s_torch, S_torch]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        N_t_torch = N_t_torch.to(device)
        B_t_torch = B_t_torch.to(device)
        N_s_torch = N_s_torch.to(device)
        B_s_torch = B_s_torch.to(device)
        S_torch = S_torch.to(device)
        
        result_torch = mh_mean_constrained_update_torch(
            N_t_torch, B_t_torch, N_s_torch, B_s_torch, S_torch, sweeps, override_support=override_support
        )
        
        # Convert back
        if primary_type == 'numpy':
            result = result_torch.cpu().numpy()
        else:
            result = result_torch
    
    return result


# no grad then compile should be fastest
@torch.no_grad
@torch.compile
def constrained_multinomial_proposal_torch(
    N_t: torch.LongTensor,     # (B,d)
    B_t: torch.LongTensor,     # (B,d)
    N_s: torch.LongTensor,     # (B,d)
    B_s: torch.LongTensor,     # (B,d)
    S:   torch.LongTensor,     # (d,)
    override_support: bool = False
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Torch implementation of constrained multinomial proposal."""
    B, d = B_s.shape
    device = B_s.device
    
    # 1) get true hyper‐geom support per‐entry
    lower = torch.maximum(torch.tensor(0),  N_s + B_t - N_t)   # (B,d)
    upper = torch.minimum(N_s, B_t)              # (B,d)

    # 2) expected under hypergeom
    N_t_float = N_t.float()
    B_s_exp = torch.where(N_t==0, 0.0, N_s.float() * (B_t.float()/N_t_float))

    # 3) column residuals & sign
    col_sum = B_s.sum(dim=0)     # (d,)
    R       = S - col_sum        # (d,)
    sgn     = torch.sign(R).long()  # (d,)
    Rabs    = torch.abs(R).long()   # (d,)

    # 4) directional weights → probs
    #    we want to favor moves that shrink |B_s - E[B_s]|
    diff    = (B_s_exp - B_s.float()) * sgn.unsqueeze(0)   # (B,d)
    w       = torch.logaddexp(torch.zeros_like(diff), diff) # (B,d)
    probs   = w / w.sum(dim=0, keepdim=True)      # (B,d)

    # 5) custom vectorizedmultinomial sampling
    p_mat = probs.T  # (d, B)
    draws = multinomial(Rabs, p_mat)
    delta = draws.T

    # 6) raw proposal and clip to true support
    B_prop = B_s + sgn.unsqueeze(0) * delta             # (B,d)

    if not override_support:
        B_prop = torch.max(torch.min(B_prop, upper), lower)

    return B_prop, delta

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

    # 2) expected under hypergeom, suppress divide by zero warnings
    safe_div = np.divide(B_t, N_t, out=np.zeros_like(B_t, dtype=np.float64), where=N_t != 0)
    B_s_exp = N_s * safe_div        
    
    #B_s_exp = np.where(N_t == 0, 0.0, N_s * (B_t / N_t))

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

    return B_prop, delta


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
        device = B_s_torch.device
        for tensor in [N_t_torch, B_t_torch, N_s_torch, S_torch]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        N_t_torch = N_t_torch.to(device)
        B_t_torch = B_t_torch.to(device)
        N_s_torch = N_s_torch.to(device)
        B_s_torch = B_s_torch.to(device)
        S_torch = S_torch.to(device)
        
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