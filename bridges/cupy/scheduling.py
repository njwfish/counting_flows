import cupy as cp


def make_weight_schedule(K, schedule_type="linear", **kwargs):
    """
    Create monotone increasing weight schedule from 0 to 1.
    
    Args:
        K: Number of steps
        schedule_type: How weights change over time
        **kwargs: Schedule-specific parameters
    
    Returns:
        weights: array of shape (K+1,) with weights from 0 to 1
    """
    # Create base time points
    w = cp.linspace(0, 1, num=K+1)
    
    if schedule_type == "linear":
        return w
    
    elif schedule_type == "cosine":
        # Cosine schedule: starts slow, accelerates, then slows
        return 0.5 * (1.0 - cp.cos(cp.pi * w))
    
    elif schedule_type == "early_dense":
        # More weight change early in the schedule
        concentration = kwargs.get('concentration', 2.0)
        return w ** (1.0 / concentration)
        
    elif schedule_type == "late_dense":
        # More weight change late in the schedule
        concentration = kwargs.get('concentration', 2.0)
        return w ** concentration
        
    elif schedule_type == "sigmoid":
        # S-shaped curve
        steepness = kwargs.get('steepness', 10.0)
        return 1.0 / (1.0 + cp.exp(-steepness * (w - 0.5)))
        
    else:
        raise ValueError(f"Unknown weight schedule: {schedule_type}") 