"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial, Beta, NegativeBinomial
from .scheduling import make_time_spacing_schedule, make_phi_schedule, get_bridge_parameters, make_lambda_schedule
from .datasets import manual_hypergeometric


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
        mode: Bridge type ("poisson", "nb", "poisson_bd", "polya_bd")
        r_min, r_max: Range for r(t) schedule in NB bridge
        r_schedule: Schedule type for r(t) ("linear", "cosine", etc.)
        time_schedule: Time spacing ("uniform", "early_dense", etc.)
        use_mean: Whether to use mean predictions instead of sampling
        device: Device for computation
        return_trajectory: Whether to return intermediate steps (for visualization)
        bd_r, bd_beta: Parameters for Polya BD bridge
        lam_p0, lam_p1, lam_m0, lam_m1: Birth/death rates for BD bridges
        bd_schedule_type: How lambda rates change over time for BD
        **schedule_kwargs: Additional parameters for schedules
    
    Returns:
        x0: Generated samples at time t=0
        trajectory: List of intermediate x values (if return_trajectory=True)
    """
    x = x1.clone().to(device).float()
    
    # Initialize trajectory storage if requested
    if return_trajectory:
        trajectory = [x.clone().cpu()]
    
    # Create schedules based on mode
    if mode == "nb":
        phi_sched, R = make_phi_schedule(K, phi_min=0.0, phi_max=r_max, schedule_type=r_schedule, **schedule_kwargs)
        phi_sched = phi_sched.to(device)
        R = R.to(device)
        # Get time points (always use make_time_spacing_schedule)
        time_points = make_time_spacing_schedule(K, time_schedule, **schedule_kwargs).to(device)
    elif mode in ["poisson_bd", "polya_bd"]:
        # BD bridges use lambda schedules
        grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
            K, lam_p0, lam_p1, lam_m0, lam_m1, bd_schedule_type, device=device
        )
        Λ_tot1 = Λp[-1] + Λm[-1]
        time_points = grid
    else:  # mode == "poisson"
        # Get time points (always use make_time_spacing_schedule)
        time_points = make_time_spacing_schedule(K, time_schedule, **schedule_kwargs).to(device)
    
    time_points = torch.flip(time_points, [0])  # Reverse for backward process
    
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
        B1 = (N + diff) // 2
    
    for step in range(K):
        t_current = time_points[step]
        t_next = time_points[step + 1] if step < K-1 else torch.tensor(0.0).to(device)
        
        t = torch.full_like(x[:, :1], t_current)
        
        # Use unified sampling interface
        x0_hat = model.sample(x, z, t, use_mean=use_mean).float()
        
        if mode in ["poisson_bd", "polya_bd"]:
            # Birth-death bridge sampling
            k_current = step
            k_next = step + 1 if step < K-1 else K
            
            # Original indexing (revert the "fix")
            w_t = (Λp[K - k_current] + Λm[K - k_current]) / Λ_tot1
            w_s = (Λp[K - k_next] + Λm[K - k_next]) / Λ_tot1 if k_next < K else torch.tensor(0.0, device=device)
            
            # Use floor instead of round to match expected value exactly
            N_t = torch.floor(N.float() * w_t).long()
            N_s = torch.floor(N.float() * w_s).long()
            
            # Recover B_t from current x and predicted x0_hat
            delta = (x - x0_hat)  # (B, d)
            
            # Validate that delta + N_t is even (required for integer B_t)
            remainder = (delta + N_t) % 2
            # if torch.any(remainder != 0):
            #     print(f"Warning: Found {torch.sum(remainder != 0)} non-even (delta + N_t) values, rounding delta")
            # Round delta to nearest even value relative to N_t
            delta = delta - remainder
            
            # Clamp delta to valid range and ensure B_t is in [0, N_t]
            delta = torch.clamp(delta, min=-N_t.float(), max=N_t.float())
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
        
        else:
            # Original one-sided bridge logic
            # Compute delta
            delta = x - x0_hat
            n = delta.abs().clamp_max(1e6).long()
            sgn = torch.sign(delta)
            
            # Bridge kernel parameters
            if mode == "nb":
                # Use proper bridge formulation from paper
                # For reverse sampling: we want the probability of decrementing by a certain amount
                
                # Get Φ(t_current) - this tells us the "amount of mass" accumulated so far
                phi_current = get_bridge_parameters(t_current, phi_sched, R)[0]  # α = Φ(t)
                phi_next = get_bridge_parameters(t_next, phi_sched, R)[0]       # α = Φ(t_next)
                
                # Step size in Φ space: how much Φ decreases in this step
                phi_step = phi_current - phi_next  # Φ(t_current) - Φ(t_next)
                
                # For reverse sampling, the probability should be proportional to 
                # the step size relative to the remaining "time mass"
                # This is a reasonable approximation to the complex reverse kernel
                
                # Correct Beta distribution parameters for reverse kernel:
                # α = Φ(s) = Φ(t_next) (target time)
                # β = Φ(t) - Φ(s) = Φ(t_current) - Φ(t_next) (step size)
                alpha_step = torch.clamp(phi_next.expand_as(n), min=1e-6)          # Φ(s) target
                beta_step = torch.clamp(phi_step.expand_as(n), min=1e-6)          # Step size
                
                p_step = Beta(alpha_step, beta_step).sample().clamp(0., 0.999).to(device)
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


def debug_bd_indexing(K=10, lam_p0=1.0, lam_p1=1.0, lam_m0=1.0, lam_m1=1.0, device="cpu"):
    """
    Debug function to verify BD bridge time indexing is correct.
    Should show that w_t goes from ~1.0 to ~0.0 as step increases.
    """
    print("=== BD Bridge Indexing Debug ===")
    
    # Create schedules
    grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
        K, lam_p0, lam_p1, lam_m0, lam_m1, "constant", device=device
    )
    Λ_tot1 = Λp[-1] + Λm[-1]
    
    print(f"Raw grid: {grid}")
    print(f"Λp: {Λp}")
    print(f"Λm: {Λm}")
    print(f"Λ_tot1: {Λ_tot1}")
    
    # Flip for reverse process
    time_points = torch.flip(grid, [0])
    print(f"Flipped time_points: {time_points}")
    
    print("\nStep -> t_current -> w_t:")
    for step in range(K):
        t_current = time_points[step]
        grid_idx_current = K - step  # Map flipped step to original grid index
        w_t = (Λp[grid_idx_current] + Λm[grid_idx_current]) / Λ_tot1
        print(f"  {step:2d} -> {t_current:.3f} -> {w_t:.3f}")
    
    print(f"\nExpected: w_t should go from {(Λp[-1] + Λm[-1]) / Λ_tot1:.3f} to {(Λp[0] + Λm[0]) / Λ_tot1:.3f}")
    print("=== End Debug ===\n")


def test_bd_linear_interpolant(model, x1, z, K=50, batch_size=1000, mode="poisson_bd", device="cpu"):
    """
    Test that BD bridge trajectories follow the expected linear interpolant in mean.
    For large batches, E[X_t] should ≈ linear interpolation between x0_hat and x1.
    """
    print("=== Testing BD Linear Interpolant ===")
    
    # Get model prediction at t=1
    t1 = torch.full_like(x1[:, :1], 1.0)
    x0_hat = model.sample(x1, z, t1, use_mean=True).float()
    
    print(f"x1 mean: {x1.float().mean():.3f}")
    print(f"x0_hat mean: {x0_hat.mean():.3f}")
    print(f"Expected trajectory: linear from {x0_hat.mean():.3f} to {x1.float().mean():.3f}")
    
    # Sample many trajectories and check means at different time points
    test_times = [0.2, 0.4, 0.6, 0.8]
    
    for t_test in test_times:
        # Generate many samples at this time point
        trajectories = []
        for _ in range(batch_size // 10):  # Process in smaller batches
            _, traj = reverse_sampler(
                x1[:10], z[:10], model, K=K, mode=mode, device=device,
                return_trajectory=True
            )
            trajectories.extend(traj)
        
        # Find trajectory point closest to t_test
        time_idx = int(t_test * K)
        x_t_samples = [traj[time_idx] for traj in trajectories]
        x_t_mean = torch.stack(x_t_samples).float().mean()
        
        # Expected linear interpolant
        expected_mean = x0_hat.mean() + t_test * (x1.float().mean() - x0_hat.mean())
        
        print(f"t={t_test:.1f}: observed={x_t_mean:.3f}, expected={expected_mean:.3f}, diff={abs(x_t_mean - expected_mean):.3f}")
    
    print("=== End Test ===\n") 