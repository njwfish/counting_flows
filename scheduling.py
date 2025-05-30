"""
Scheduling System for Count-based Flow Matching

Provides two separate scheduling functions:
1. r(t) parameter scheduling for NB bridges 
2. Time point spacing for bridge sampling
"""

import torch


def make_r_schedule(K, r_min=1.0, r_max=20.0, schedule_type="linear", **kwargs):
    """
    Create r(t) schedule for Negative Binomial bridges.
    
    Args:
        K: Number of steps
        r_min, r_max: Range for r values
        schedule_type: How r changes over time
        **kwargs: Schedule-specific parameters
    
    Returns:
        r_values: tensor of shape (K+1,) with r values from r_max to r_min
    """
    steps = torch.arange(K+1, dtype=torch.float32)
    t = steps / K  # Time points from 0 to 1
    
    if schedule_type == "linear":
        r_values = r_max + (r_min - r_max) * t
        
    elif schedule_type == "cosine":
        # Cosine schedule (smooth transitions)
        cosine_weights = 0.5 * (1 + torch.cos(torch.pi * t))
        r_values = r_min + (r_max - r_min) * cosine_weights
        
    elif schedule_type == "exponential":
        # Exponential decay
        decay_rate = kwargs.get('decay_rate', 2.0)
        exp_weights = torch.exp(-decay_rate * t)
        r_values = r_min + (r_max - r_min) * exp_weights
        
    elif schedule_type == "sigmoid":
        # Sigmoid transition (sharp change in middle)
        steepness = kwargs.get('steepness', 10.0)
        midpoint = kwargs.get('midpoint', 0.5)
        sigmoid_weights = torch.sigmoid(-steepness * (t - midpoint))
        r_values = r_min + (r_max - r_min) * sigmoid_weights
        
    elif schedule_type == "polynomial":
        # Polynomial schedule
        power = kwargs.get('power', 2.0)
        poly_weights = (1 - t) ** power
        r_values = r_min + (r_max - r_min) * poly_weights
        
    elif schedule_type == "sqrt":
        # Square root schedule (fast initial decay, slow later)
        sqrt_weights = torch.sqrt(1 - t)
        r_values = r_min + (r_max - r_min) * sqrt_weights
        
    elif schedule_type == "inverse_sqrt":
        # Inverse square root (slow initial decay, fast later)
        eps = 1e-8
        inv_sqrt_weights = 1.0 / torch.sqrt(t + eps)
        inv_sqrt_weights = inv_sqrt_weights / inv_sqrt_weights[0]  # Normalize
        r_values = r_min + (r_max - r_min) * inv_sqrt_weights
        
    else:
        raise ValueError(f"Unknown r schedule type: {schedule_type}")
    
    return r_values


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
