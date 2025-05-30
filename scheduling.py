"""
Time Scheduling System for Count-based Flow Matching

Provides flexible scheduling functions for both r(t) parameters in NB bridges
and non-uniform time spacing for bridge sampling.
"""

import torch


def make_time_schedule(K, schedule_type="linear", **kwargs):
    """
    Unified time scheduling system for bridges.
    
    Args:
        K: Number of steps (returns K+1 values for t=0 to t=1)
        schedule_type: Type of schedule
        **kwargs: Schedule-specific parameters
    
    Returns:
        times: tensor of shape (K+1,) with values from 0 to 1
        weights: tensor of shape (K+1,) with schedule-specific weights
    """
    steps = torch.arange(K+1, dtype=torch.float32)
    times = steps / K  # Uniform time grid from 0 to 1
    
    if schedule_type == "linear":
        # Linear interpolation between start and end values
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        weights = start_val + (end_val - start_val) * times
        
    elif schedule_type == "cosine":
        # Cosine schedule (smooth transitions)
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        cosine_weights = 0.5 * (1 + torch.cos(torch.pi * times))
        weights = end_val + (start_val - end_val) * cosine_weights
        
    elif schedule_type == "exponential":
        # Exponential decay
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        decay_rate = kwargs.get('decay_rate', 2.0)
        weights = start_val * torch.exp(-decay_rate * times)
        weights = torch.clamp(weights, min=end_val)
        
    elif schedule_type == "sigmoid":
        # Sigmoid transition (sharp change in middle)
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        steepness = kwargs.get('steepness', 10.0)
        midpoint = kwargs.get('midpoint', 0.5)
        sigmoid_weights = torch.sigmoid(-steepness * (times - midpoint))
        weights = end_val + (start_val - end_val) * sigmoid_weights
        
    elif schedule_type == "polynomial":
        # Polynomial schedule
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        power = kwargs.get('power', 2.0)
        poly_weights = (1 - times) ** power
        weights = end_val + (start_val - end_val) * poly_weights
        
    elif schedule_type == "sqrt":
        # Square root schedule (fast initial decay, slow later)
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        sqrt_weights = torch.sqrt(1 - times)
        weights = end_val + (start_val - end_val) * sqrt_weights
        
    elif schedule_type == "inverse_sqrt":
        # Inverse square root (slow initial decay, fast later)
        start_val = kwargs.get('start_val', 1.0)
        end_val = kwargs.get('end_val', 0.1)
        # Avoid division by zero at t=1
        eps = 1e-8
        inv_sqrt_weights = 1.0 / torch.sqrt(times + eps)
        inv_sqrt_weights = inv_sqrt_weights / inv_sqrt_weights[0]  # Normalize
        weights = end_val + (start_val - end_val) * inv_sqrt_weights
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return times, weights


def make_r_schedule(K, r_min=1.0, r_max=20.0, scheme="linear", **kwargs):
    """
    Backward compatibility wrapper for r schedules.
    """
    times, weights = make_time_schedule(
        K, schedule_type=scheme, 
        start_val=r_max, end_val=r_min, 
        **kwargs
    )
    return weights


def make_time_spacing_schedule(K, schedule_type="uniform", **kwargs):
    """
    Create non-uniform time spacing for bridges.
    
    Returns:
        times: tensor of shape (K+1,) with non-uniform spacing
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