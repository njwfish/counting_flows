"""
Scheduling System for Count-based Flow Matching

Provides three separate scheduling functions:
1. Cumulative shape function Φ(t) scheduling for NB bridges (monotone increasing)
2. Time point spacing for bridge sampling
3. Lambda schedules for birth-death bridges with birth/death rates
"""

import torch
import numpy as np


def make_phi_schedule(K, phi_min=0., phi_max=20.0, schedule_type="linear", **kwargs):
    """
    Create cumulative shape function Φ(t) schedule directly.
    
    This implements the proper bridge formulation from the paper:
    - α_t = Φ(t) 
    - β_t = R - Φ(t) where R = Φ(1) = phi_max
    
    Φ(t) is monotone increasing from phi_min to phi_max.
    
    Args:
        K: Number of steps
        phi_min, phi_max: Range for Φ values (typically phi_min=0)
        schedule_type: How Φ changes over time
        **kwargs: Schedule-specific parameters
    
    Returns:
        phi_values: tensor of shape (K+1,) with Φ(t) values from phi_min to phi_max
        R: scalar value Φ(1) = phi_max
    """
    steps = torch.arange(K+1, dtype=torch.float32)
    t = steps / K  # Time points from 0 to 1
    
    if schedule_type == "linear":
        phi_values = phi_min + (phi_max - phi_min) * t
        
    elif schedule_type == "cosine":
        # Cosine schedule (smooth transitions)
        # Use 1 - cos to get increasing function
        cosine_weights = 0.5 * (1 - torch.cos(torch.pi * t))
        phi_values = phi_min + (phi_max - phi_min) * cosine_weights
        
    elif schedule_type == "exponential":
        # Exponential growth
        growth_rate = kwargs.get('growth_rate', 2.0)
        exp_weights = (torch.exp(growth_rate * t) - 1) / (torch.exp(growth_rate) - 1)
        phi_values = phi_min + (phi_max - phi_min) * exp_weights
        
    elif schedule_type == "sigmoid":
        # Sigmoid transition (sharp change in middle)
        steepness = kwargs.get('steepness', 10.0)
        midpoint = kwargs.get('midpoint', 0.5)
        sigmoid_weights = torch.sigmoid(steepness * (t - midpoint))
        # Normalize to [0, 1]
        sigmoid_weights = (sigmoid_weights - sigmoid_weights[0]) / (sigmoid_weights[-1] - sigmoid_weights[0])
        phi_values = phi_min + (phi_max - phi_min) * sigmoid_weights
        
    elif schedule_type == "polynomial":
        # Polynomial schedule
        power = kwargs.get('power', 2.0)
        poly_weights = t ** power
        phi_values = phi_min + (phi_max - phi_min) * poly_weights
        
    elif schedule_type == "sqrt":
        # Square root schedule (fast initial growth, slow later)
        sqrt_weights = torch.sqrt(t)
        phi_values = phi_min + (phi_max - phi_min) * sqrt_weights
        
    elif schedule_type == "log":
        # Logarithmic schedule (slow initial growth, faster later)
        eps = 1e-8
        log_weights = torch.log(t + eps) - torch.log(torch.tensor(eps))
        log_weights = log_weights / log_weights[-1]  # Normalize to [0, 1]
        phi_values = phi_min + (phi_max - phi_min) * log_weights
        
    else:
        raise ValueError(f"Unknown phi schedule type: {schedule_type}")
    
    # Ensure monotonicity and correct endpoints
    phi_values[0] = phi_min
    phi_values[-1] = phi_max
    
    # Ensure monotone increasing
    for i in range(1, len(phi_values)):
        phi_values[i] = torch.max(phi_values[i], phi_values[i-1])
    
    R = phi_values[-1]  # Φ(1) = phi_max
    
    return phi_values, R


def get_bridge_parameters(t, phi_schedule, R):
    """
    Get bridge parameters α_t and β_t for a given time t.
    
    Args:
        t: Time value(s) in [0, 1]
        phi_schedule: Φ(t) values at discretized time points
        R: Total cumulative shape Φ(1)
    
    Returns:
        alpha_t: Φ(t) values
        beta_t: R - Φ(t) values
    """
    # Convert continuous time t to discrete indices
    K = len(phi_schedule) - 1
    t_clamped = torch.clamp(t, 0.0, 1.0)
    indices = (t_clamped * K).long()
    indices = torch.clamp(indices, 0, K-1)
    
    # Look up Φ(t) values
    alpha_t = phi_schedule[indices]
    beta_t = R - alpha_t
    
    # Ensure positive values
    alpha_t = torch.clamp(alpha_t, min=1e-6)
    beta_t = torch.clamp(beta_t, min=1e-6)
    
    return alpha_t, beta_t


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
    K: int,
    lam_p0: float = 8.0,
    lam_p1: float = 8.0,
    lam_m0: float = 8.0,
    lam_m1: float = 8.0,
    schedule_type: str = "constant",
    device="cpu",
):
    """
    Build λ⁺(t_k), λ⁻(t_k) on the grid t_k = k/K  (k = 0…K)
    together with their cumulative integrals Λ⁺(t_k), Λ⁻(t_k).

    schedule_type ∈ {"constant","linear","cosine"} applied
    independently to birth (+) and death (−) rates.
    
    Args:
        K: Number of steps
        lam_p0, lam_p1: Birth rate at t=0 and t=1
        lam_m0, lam_m1: Death rate at t=0 and t=1
        schedule_type: How rates change over time
        device: Device for computation
    
    Returns:
        grid: Time points (K+1,)
        lam_plus: Birth rates λ⁺(t) (K+1,)
        lam_minus: Death rates λ⁻(t) (K+1,)
        Λp: Cumulative birth integral (K+1,)
        Λm: Cumulative death integral (K+1,)
    """
    grid = torch.linspace(0.0, 1.0, K + 1, device=device)

    def _interp(u, lo, hi):
        if schedule_type == "linear":
            return lo + (hi - lo) * u
        if schedule_type == "cosine":
            return lo + 0.5 * (hi - lo) * (1.0 - torch.cos(torch.pi * u))
        return torch.full_like(u, lo)

    lam_plus  = _interp(grid, lam_p0, lam_p1)
    lam_minus = _interp(grid, lam_m0, lam_m1)

    # trapezoidal cum‑integral ∫ λ(t) dt
    dt = grid[1:] - grid[:-1]
    cum_plus  = torch.cumsum(0.5 * (lam_plus[1:]  + lam_plus[:-1])  * dt, dim=0)
    cum_minus = torch.cumsum(0.5 * (lam_minus[1:] + lam_minus[:-1]) * dt, dim=0)

    Λp = torch.cat([torch.zeros(1, device=device), cum_plus])
    Λm = torch.cat([torch.zeros(1, device=device), cum_minus])
    return grid, lam_plus, lam_minus, Λp, Λm
 