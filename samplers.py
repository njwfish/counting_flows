"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial, Beta, NegativeBinomial
from .scheduling import make_time_spacing_schedule, make_phi_schedule, get_bridge_parameters, make_lambda_schedule
from .bridges import manual_hypergeometric


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
    return_x_hat=False,
    # BD-specific parameters
    bd_r=1.0,
    bd_beta=1.0,
    lam_p0=8.0,
    lam_p1=8.0,
    lam_m0=8.0,
    lam_m1=8.0,
    bd_schedule_type="constant",
    **schedule_kwargs
):
    """
    Unified reverse sampler that works with any BaseCountModel
    
    Args:
        x1: Starting point (typically from target distribution)
        z: Context vector [λ₀, λ₁]
        model: Trained neural network model
        K: Number of reverse steps
        mode: Bridge type ("poisson", "nb", "poisson_bd", "polya_bd", "reflected_bd")
        r_min, r_max: Range for r(t) schedule in NB bridge
        r_schedule: Schedule type for r(t) ("linear", "cosine", etc.)
        time_schedule: Time spacing ("uniform", "early_dense", etc.)
        use_mean: Whether to use mean predictions instead of sampling
        device: Device for computation
        return_trajectory: Whether to return intermediate steps (for visualization)
        return_x_hat: Whether to return x0_hat predictions at each step (when return_trajectory=True)
        bd_r, bd_beta: Parameters for Polya BD bridge
        lam_p0, lam_p1, lam_m0, lam_m1: Birth/death rates for BD bridges
        bd_schedule_type: How lambda rates change over time for BD
        lam0, lam1: Birth/death rates for reflected BD bridge (equal rates)
        **schedule_kwargs: Additional parameters for schedules
    
    Returns:
        x0: Generated samples at time t=0
        trajectory: List of intermediate x values (if return_trajectory=True)
        x_hat_trajectory: List of x0_hat predictions at each step (if return_trajectory=True and return_x_hat=True)
    """
    
    x = x1.clone().to(device).float()
    
    # Initialize trajectory storage if requested
    if return_trajectory:
        trajectory = [x.clone().cpu()]
        if return_x_hat:
            x_hat_trajectory = []
    

    grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
        K, lam_p0, lam_p1, lam_m0, lam_m1, bd_schedule_type, device=device
    )
    Λ_tot1 = Λp[-1] + Λm[-1]
    time_points = grid
    
    time_points = torch.flip(time_points, [0])
    
    # For BD bridges, we need to sample the latent variables N, B1 at the start
    if mode in ["poisson_bd", "polya_bd"]:
        # Predict x0_hat at t=1 to get diff
        t1 = torch.full_like(x[:, :1], 1.0)
        x0_hat = model.sample(x.float(), z, t1, use_mean=use_mean).float()
        diff = (x - x0_hat)  # (B, d) - don't squeeze
        
        # Sample latent variables
        if mode == "polya_bd":
            # Use NegativeBinomial
            nb = NegativeBinomial(total_count=bd_r, probs=bd_beta / (bd_beta + 1))
            M = nb.sample(diff.shape).long().to(device)
        else:  # poisson_bd - must reference λ‑schedule!
            # Use the cumulative lambda integrals
            Lambda_p = Λp[-1]  # Total birth integral
            Lambda_m = Λm[-1]  # Total death integral
            lambda_star = 2.0 * torch.sqrt(Lambda_p * Lambda_m)        # scalar
            lambda_star = lambda_star.unsqueeze(-1).expand_as(diff)    # shape (B, d)
            M = torch.poisson(lambda_star).long().to(device)
        
        N = diff.abs() + 2 * M
    
    for step in range(K):
        t_current = time_points[step]
        t_next = time_points[step + 1] if step < K-1 else torch.tensor(0.0).to(device)
        
        t = torch.full_like(x[:, :1], t_current)
        
        # Use unified sampling interface
        x0_hat = model.sample(x, z, t, use_mean=use_mean).float()
        print(x1.float().mean(0), x.float().mean(0), x0_hat.float().mean(0), z.float().mean(0), t.float().mean())
        
        # Store x0_hat prediction if requested
        if return_trajectory and return_x_hat:
            x_hat_trajectory.append(x0_hat.clone().cpu())
        
        # Birth-death bridge sampling
        k_current = step
        k_next = step + 1 if step < K-1 else K
        
        # Original indexing (revert the "fix")
        w_t = (Λp[K - k_current] + Λm[K - k_current]) / Λ_tot1
        w_s = (Λp[K - k_next] + Λm[K - k_next]) / Λ_tot1 if k_next < K else torch.tensor(0.0, device=device)
        
        # Use floor instead of round to match expected value exactly
        if step == 0:
            N_t = torch.binomial(N.float(), w_t).long() #  torch.floor(N.float() * w_t).long()
        else:
            N_t = N_s
        N_s = torch.binomial(N_t.float(), w_s / w_t).long() #  torch.floor(N.float() * w_s).long()
        
        # Recover B_t from current x and predicted x0_hat
        delta = (x - x0_hat)  # (B, d)
        
        # Validate that delta + N_t is even (required for integer B_t)
        remainder = (delta + N_t) % 2
        # if torch.any(remainder != 0):
        #     print(f"Warning: Found {torch.sum(remainder != 0)} non-even (delta + N_t) values, rounding delta")
        # Round delta to nearest even value relative to N_t
        delta = delta - remainder
        
        # Clamp delta to valid range and ensure B_t is in [0, N_t]
        # delta = torch.clamp(delta, min=-N_t.float(), max=N_t.float())
        B_t = ((N_t + delta) // 2).long()
        # Ensure B_t is in valid range [0, N_t] element-wise
        B_t = torch.maximum(torch.zeros_like(B_t), torch.minimum(B_t, N_t))
        
        # Draw B_s ~ Hypergeom(N_t, B_t, N_s)
        B_s = torch.zeros_like(B_t)
        
        # Vectorized hypergeometric sampling
        B_s_np = manual_hypergeometric(
            total_count=N_t.cpu().numpy(),
            num_successes=B_t.cpu().numpy(), 
            num_draws=N_s.cpu().numpy()
        )
        B_s = torch.from_numpy(B_s_np).to(device)
        
        # Update x: x_{s_k} = x_t - 2*(B_t - B_s) + (N_t - N_s)
        x = x - 2 * (B_t - B_s) + (N_t - N_s)
    
        # Store trajectory step if requested
        if return_trajectory:
            trajectory.append(x.clone().cpu())
    
    if return_trajectory:
        if return_x_hat:
            return x.long(), trajectory, x_hat_trajectory
        else:
            return x.long(), trajectory
    else:
        return x.long()
