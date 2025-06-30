import numpy as np

def make_time_spacing_schedule(K, schedule_type="uniform", **kwargs):
    """
    Create time point spacing for bridge sampling.
    
    Args:
        K: Number of steps
        schedule_type: How time points are distributed
        **kwargs: Schedule-specific parameters
    
    Returns:
        times: array of shape (K+1,) with time points from 0 to 1
    """
    if schedule_type == "uniform":
        return np.linspace(0, 1, K+1)
    
    elif schedule_type == "early_dense":
        concentration = kwargs.get('concentration', 2.0)
        steps = np.linspace(0, 1, K+1)
        times = steps ** concentration
        return times
        
    elif schedule_type == "late_dense":
        concentration = kwargs.get('concentration', 2.0)
        steps = np.linspace(0, 1, K+1)
        times = 1.0 - (1.0 - steps) ** concentration
        return times
        
    elif schedule_type == "middle_dense":
        concentration = kwargs.get('concentration', 2.0)
        steps = np.linspace(0, 1, K+1)
        sine_input = steps * np.pi
        sine_values = np.sin(sine_input)
        times = (sine_values / sine_values.max()) ** (1.0 / concentration)
        times[0] = 0.0
        times[-1] = 1.0
        return times
    
    else:
        raise ValueError(f"Unknown time spacing schedule: {schedule_type}")


def make_lambda_schedule(
    timepoints: np.ndarray,
    lam0: float = 8.0,
    lam1: float = 8.0,
    schedule_type: str = "constant",
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
            return lo + 0.5 * (hi - lo) * (1.0 - np.cos(np.pi * u))
        return np.full_like(u, lo)

    lam_plus  = _interp(timepoints, lam0, lam1)
    lam_minus = _interp(timepoints, lam0, lam1)

    # trapezoidal cum‑integral ∫ λ(t) dt
    dt = timepoints[1:] - timepoints[:-1]
    cum_plus  = np.cumsum(0.5 * (lam_plus[1:]  + lam_plus[:-1])  * dt, axis=0)
    cum_minus = np.cumsum(0.5 * (lam_minus[1:] + lam_minus[:-1]) * dt, axis=0)

    Λp = np.concatenate([np.zeros(1), cum_plus])
    Λm = np.concatenate([np.zeros(1), cum_minus])
    return lam_plus, lam_minus, Λp, Λm 