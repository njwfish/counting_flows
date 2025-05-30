"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial, Beta
from .scheduling import make_r_schedule, make_time_spacing_schedule


@torch.no_grad()
def reverse_sampler(
    x1, z, model, *,
    K=50, mode="poisson", 
    r_min=1.0, r_max=20.0,
    r_schedule="linear",
    time_schedule="uniform",
    use_mean=False,
    device="cuda",
    return_trajectory=False,
    **schedule_kwargs
):
    """
    Unified reverse sampler that works with any BaseCountModel
    
    Args:
        x1: Starting point (typically from target distribution)
        z: Context vector [λ₀, λ₁]
        model: Trained neural network model
        K: Number of reverse steps
        mode: Bridge type ("poisson" or "nb")
        r_min, r_max: Range for r(t) schedule in NB bridge
        r_schedule: Schedule type for r(t) ("linear", "cosine", etc.)
        time_schedule: Time spacing ("uniform", "early_dense", etc.)
        use_mean: Whether to use mean predictions instead of sampling
        device: Device for computation
        return_trajectory: Whether to return intermediate steps (for visualization)
        **schedule_kwargs: Additional parameters for schedules
    
    Returns:
        x0: Generated samples at time t=0
        trajectory: List of intermediate x values (if return_trajectory=True)
    """
    x = x1.clone().to(device).float()
    
    # Initialize trajectory storage if requested
    if return_trajectory:
        trajectory = [x.clone().cpu()]
    
    # Create schedules
    if mode == "nb":
        r_sched = make_r_schedule(K, r_min, r_max, r_schedule, **schedule_kwargs).to(device)
    
    # Get time points (always use make_time_spacing_schedule)
    time_points = make_time_spacing_schedule(K, time_schedule, **schedule_kwargs).to(device)
    time_points = torch.flip(time_points, [0])  # Reverse for backward process
    
    for step in range(K):
        t_current = time_points[step]
        t_next = time_points[step + 1] if step < K-1 else torch.tensor(0.0).to(device)
        
        t = torch.full_like(x[:, :1], t_current)
        
        # Use unified sampling interface
        x0_hat = model.sample(x, z, t, use_mean=use_mean).float()
        
        # Compute delta
        delta = x - x0_hat
        n = delta.abs().clamp_max(1e6).long()
        sgn = torch.sign(delta)
        
        # Bridge kernel parameters
        if mode == "nb":
            k_idx = K - step  # Convert to forward index for r_sched
            r_k = r_sched[k_idx].item() if k_idx < len(r_sched) else r_sched[-1].item()
            
            # Time remaining and step size
            dt_step = t_current - t_next
            alpha = max(r_k * t_next.item(), 1e-6)  # Time remaining
            beta = max(r_k * dt_step.item(), 1e-6)  # Step size
            p_step = Beta(alpha, beta).sample(n.shape).clamp(0., 0.999).to(device)
        else:  # Poisson
            dt_step = t_current - t_next
            p_step = torch.full_like(x, dt_step / t_current).clamp(0., 0.999)
        
        # Sample decrements only where n > 0
        krem = torch.zeros_like(n).float().to(device)
        mask = n > 0
        
        if mask.any():
            krem[mask] = Binomial(
                total_count=n[mask],
                probs=p_step[mask]
            ).sample()
        
        # Apply step
        x = x - sgn * krem
        
        # Store trajectory step if requested
        if return_trajectory:
            trajectory.append(x.clone().cpu())
    
    if return_trajectory:
        return x.long(), trajectory
    else:
        return x.long() 