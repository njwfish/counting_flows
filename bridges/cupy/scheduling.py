import cupy as cp

weight_functions = {
    "linear": lambda w: w,
    "cosine": lambda w: 0.5 * (1.0 - cp.cos(cp.pi * w)),
    "early_dense": lambda w, **kwargs: w ** (1.0 / kwargs.get('concentration', 2.0)),
    "late_dense": lambda w, **kwargs: w ** kwargs.get('concentration', 2.0),
    "sigmoid": lambda w, **kwargs: 1.0 / (1.0 + cp.exp(-kwargs.get('steepness', 10.0) * (w - 0.5))),
}

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
    
    return weight_functions[schedule_type](w, **kwargs)
