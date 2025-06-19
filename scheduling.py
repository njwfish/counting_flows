"""
Scheduling System for Count-based Flow Matching

Provides three separate scheduling functions:
1. Cumulative shape function Φ(t) scheduling for NB bridges (monotone increasing)
2. Time point spacing for bridge sampling
3. Lambda schedules for birth-death bridges with birth/death rates
"""

import torch
import numpy as np

def make_time_spacing_schedule(K, schedule_type="uniform", **kwargs):
    """
    Create time point spacing for bridge sampling.
    
    Args:
        K: Number of steps
        schedule_type: How time points are distributed
        **kwargs: Schedule-specific parameters
    
    Returns:
        times: tensor of shape (K+1,) with time points from 0 to 1
    """
    if schedule_type == "uniform":
        return torch.linspace(0, 1, K+1)
    
    elif schedule_type == "early_dense":
        # More steps early in the process using power function
        concentration = kwargs.get('concentration', 2.0)
        steps = torch.linspace(0, 1, K+1)
        # Use power function to concentrate early (smaller values get more steps)
        times = steps ** concentration
        return times
        
    elif schedule_type == "late_dense":
        # More steps late in the process
        concentration = kwargs.get('concentration', 2.0)
        steps = torch.linspace(0, 1, K+1)
        # Reverse power function to concentrate late
        times = 1.0 - (1.0 - steps) ** concentration
        return times
        
    elif schedule_type == "middle_dense":
        # More steps in the middle using sine function
        concentration = kwargs.get('concentration', 2.0)
        steps = torch.linspace(0, 1, K+1)
        # Use sine to concentrate in middle
        # Transform to [0, π] then apply sine and normalize
        sine_input = steps * torch.pi
        sine_values = torch.sin(sine_input)
        # Normalize to [0, 1] and apply concentration
        times = (sine_values / sine_values.max()) ** (1.0 / concentration)
        # Ensure endpoints are correct
        times[0] = 0.0
        times[-1] = 1.0
        return times
    
    else:
        raise ValueError(f"Unknown time spacing schedule: {schedule_type}")


def make_lambda_schedule(
    timepoints: torch.Tensor,
    lam0: float = 8.0,
    lam1: float = 8.0,
    schedule_type: str = "constant",
    device="cpu",
):
    """
    Build λ⁺(t_k), λ⁻(t_k) on the grid t_k = k/K  (k = 0…K)
    together with their cumulative integrals Λ⁺(t_k), Λ⁻(t_k).

    schedule_type ∈ {"constant","linear","cosine"} applied
    independently to birth (+) and death (−) rates.
    
    Args:
        timepoints: Time points (K+1,)
        lam0, lam1: Birth rate at t=0 and t=1
        schedule_type: How rates change over time
        device: Device for computation
    
    Returns:
        lam_plus: Birth rates λ⁺(t) (K+1,)
        lam_minus: Death rates λ⁻(t) (K+1,)
        Λp: Cumulative birth integral (K+1,)
        Λm: Cumulative death integral (K+1,)
    """
    

    def _interp(u, lo, hi):
        if schedule_type == "linear":
            return lo + (hi - lo) * u
        if schedule_type == "cosine":
            return lo + 0.5 * (hi - lo) * (1.0 - torch.cos(torch.pi * u))
        return torch.full_like(u, lo)

    lam_plus  = _interp(timepoints, lam0, lam1)
    lam_minus = _interp(timepoints, lam0, lam1)

    # trapezoidal cum‑integral ∫ λ(t) dt
    dt = timepoints[1:] - timepoints[:-1]
    cum_plus  = torch.cumsum(0.5 * (lam_plus[1:]  + lam_plus[:-1])  * dt, dim=0)
    cum_minus = torch.cumsum(0.5 * (lam_minus[1:] + lam_minus[:-1]) * dt, dim=0)

    Λp = torch.cat([torch.zeros(1, device=device), cum_plus])
    Λm = torch.cat([torch.zeros(1, device=device), cum_minus])
    return lam_plus, lam_minus, Λp, Λm
 