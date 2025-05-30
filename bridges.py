"""
Bridge Sampling Functions for Count-based Flow Matching

Provides bridge samplers for different count distributions:
- Poisson Bridge: Exact bridge between Poisson(λ₀) and Poisson(λ₁)
- Negative Binomial Bridge: Polya bridge with time-varying dispersion
"""

import torch
from torch.distributions import Poisson, Binomial, Beta
from .scheduling import make_r_schedule, make_time_spacing_schedule


def sample_batch_poisson(B, d, n_steps, lam_scale=50., time_schedule="uniform", **schedule_kwargs):
    """Sample batch using Poisson bridge with flexible time scheduling"""
    lam0, lam1 = lam_scale*torch.rand(B, d), lam_scale*torch.rand(B, d)
    x0, x1     = Poisson(lam0).sample(), Poisson(lam1).sample()
    
    # Use flexible time scheduling
    if time_schedule == "uniform":
        ts = torch.randint(1, n_steps-1, (B, 1)) / n_steps
    else:
        time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)
        # Randomly select from non-uniform time points (excluding endpoints)
        valid_indices = torch.randint(1, len(time_points)-1, (B,))  # Remove extra dimension
        ts = time_points[valid_indices].unsqueeze(-1)  # Shape (B, 1)
    
    # Exact Poisson bridge state
    n   = (x1 - x0).abs().long()
    k   = Binomial(total_count=n, probs=ts).sample()
    x_t = x0 + torch.sign(x1 - x0) * k
    z   = torch.cat([lam0, lam1], dim=1)
    
    return x0, x1, x_t, ts, z, None


def sample_batch_nb(B, d, n_steps, r_min=1.0, r_max=20., lam_scale=10., 
                   r_schedule="linear", time_schedule="uniform", **schedule_kwargs):
    """Sample batch using Negative Binomial (Polya) bridge with flexible scheduling"""
    
    # Create r(t) schedule with new system
    r_sched = make_r_schedule(n_steps, r_min, r_max, r_schedule, **schedule_kwargs)
    
    lam0, lam1 = lam_scale*torch.rand(B, d), lam_scale*torch.rand(B, d)
    x0, x1     = Poisson(lam0).sample(), Poisson(lam1).sample()
    
    # Use flexible time scheduling  
    if time_schedule == "uniform":
        ks = torch.randint(1, n_steps-1, (B, 1))
        ts = ks / n_steps
        k  = ks.squeeze(-1)
    else:
        time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)
        valid_indices = torch.randint(1, len(time_points)-1, (B,))  # Remove extra dimension
        ts = time_points[valid_indices].unsqueeze(-1)  # Shape (B, 1) 
        # Convert back to discrete indices for r_sched lookup
        k = (ts.squeeze(-1) * n_steps).round().long().clamp(0, n_steps-1)  # Fix bounds
    
    r_t = r_sched[k]
    
    n = (x1 - x0).abs().long()
    alpha = r_t.view(-1, 1) * ts
    beta  = r_t.view(-1, 1) * (1 - ts)
    alpha = alpha.clamp_min(1e-3)
    beta  = beta.clamp_min(1e-3)
    
    p   = Beta(alpha, beta).sample()
    k   = Binomial(total_count=n, probs=p).sample()
    x_t = x0 + torch.sign(x1 - x0) * k
    z   = torch.cat([lam0, lam1], dim=1)
    
    return x0, x1, x_t, ts, z, r_t.view(-1, 1).expand(B, d) 