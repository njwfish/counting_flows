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
    # Reflected BD parameters
    lam0=8.0,
    lam1=8.0,
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
        bd_r, bd_beta: Parameters for Polya BD bridge
        lam_p0, lam_p1, lam_m0, lam_m1: Birth/death rates for BD bridges
        bd_schedule_type: How lambda rates change over time for BD
        lam0, lam1: Birth/death rates for reflected BD bridge (equal rates)
        **schedule_kwargs: Additional parameters for schedules
    
    Returns:
        x0: Generated samples at time t=0
        trajectory: List of intermediate x values (if return_trajectory=True)
    """
    # Handle reflected BD bridge separately
    if mode == "reflected_bd":
        return reverse_sampler_reflected_bd(
            x1, z, model,
            n_steps=K,
            lam0=lam0,
            lam1=lam1,
            use_mean=use_mean,
            device=device,
            return_trajectory=return_trajectory
        )
    
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


@torch.no_grad()
def reverse_sampler_reflected_bd(
    x1, z, model,
    n_steps=50,
    lam0=8.0, lam1=8.0,
    use_mean=False,
    device="cuda",
    return_trajectory=False
):
    """
    Reverse sampler for the reflected birth–death bridge (λ₊=λ₋).

    Args:
      x1:   (B,d)  endpoint at t=1 (nonnegative)
      z:    (B,d)  context (passed to model but not used here)
      model: neural net with method `sample(x, t, z, use_mean)`
             returning a Tensor (B,d) of predicted hat_x0.
      n_steps: number of discretization steps K
      lam0, lam1: birth=death rates at t=0 and t=1 (equal for +/-)
      use_mean: if True, use deterministic prediction; else sample distribution
      device: "cuda" or "cpu"
      return_trajectory: if True, return list of all x_t
    Returns:
      x0:        (B,d)  final sample at t=0
      trajectory: list of (B,d) states at each time step (only if return_trajectory)
    """
    B, d = x1.shape
    x = x1.clone().to(device).long()  # current reflected state at t=1

    # 1) Build time grid and λ(t_k), Λ(t_k)
    t_grid = torch.linspace(0, 1, steps=n_steps, device=device)  # (K,)
    lam_vals = lam0 + (lam1 - lam0) * t_grid                      # (K,)
    dt = 1.0 / (n_steps - 1)
    lam_mid = 0.5 * (lam_vals[:-1] + lam_vals[1:])
    Lam = torch.zeros_like(lam_vals)
    Lam[1:] = torch.cumsum(lam_mid * dt, dim=0)  # (K,)
    Lambda1 = Lam[-1]                            # scalar

    if return_trajectory:
        trajectory = [x.clone().cpu()]

    # 2) initial prediction of x0 at t=1
    t1 = torch.ones(B, 1, device=device)  # (B,1)
    x0_hat = model.sample(x.float(), z, t1, use_mean=use_mean).long()  # (B,d)

    # 3) latent N,B1 at t=1
    diff = (x - x0_hat).abs()                   # (B,d)
    lam_star = 2.0 * Lambda1                    # scalar
    M = torch.poisson(lam_star * torch.ones_like(diff, dtype=torch.float32)).long().to(device)  # (B,d)
    N = diff + 2 * M                             # (B,d)
    B1 = (N + (x - x0_hat)) // 2                  # (B,d)

    # 4) backward loop over k = K, K-1, …, 1
    for k in range(n_steps - 1, 0, -1):
        # 4a) compute N_t = floor(N * w(t_k)),  w(t_k)=Λ(t_k)/Λ(1)
        w_k = (Lam[k] / Lambda1).clamp(0.0, 1.0)           # scalar
        N_t = torch.floor(N.float() * w_k).long()          # (B,d)

        # 4b) compute candidate signed‐parent births B_t^(+) and B_t^(-)
        #     given observed |X_t| = x
        #     a = x0_hat  (predicted "a" at this step)
        a = x0_hat  # (B,d)
        b_plus  = (N_t + (x - a)) // 2  # (B,d)
        b_minus = (N_t - (x - a)) // 2  # (B,d)

        # 4c) evaluate Hypergeom PMF at b_plus and b_minus (unnormalised)
        #     h_plus = C(B1, b_plus) * C(N - B1, N_t - b_plus) / C(N, N_t)
        #     h_minus = C(B1, b_minus) * C(N - B1, N_t - b_minus) / C(N, N_t)
        # We only need the *ratio*, so denominator C(N,N_t) cancels.

        # For numerical stability, compute in log-space:
        # log_h = log C(B1, b) + log C(N - B1, N_t - b)
        # where log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        N_f  = N.float()
        B1_f = B1.float()
        Nt_f = N_t.float()

        # clamp arguments to valid range before computing lgamma
        b_plus_clamped  = torch.maximum(torch.zeros_like(b_plus), torch.minimum(b_plus, N))   # (B,d)
        b_minus_clamped = torch.maximum(torch.zeros_like(b_minus), torch.minimum(b_minus, N))  # (B,d)

        # log C(B1, b_plus)
        logC1_plus  = (
            torch.lgamma(B1_f + 1.0)
            - torch.lgamma(b_plus_clamped.float() + 1.0)
            - torch.lgamma((B1_f - b_plus_clamped.float()) + 1.0)
        )
        # log C(N - B1, N_t - b_plus)
        logC2_plus  = (
            torch.lgamma((N_f - B1_f) + 1.0)
            - torch.lgamma((N_t.float() - b_plus_clamped.float()) + 1.0)
            - torch.lgamma(((N_f - B1_f) - (N_t.float() - b_plus_clamped.float())) + 1.0)
        )
        log_h_plus = logC1_plus + logC2_plus  # (B,d)

        # similarly for b_minus
        logC1_minus = (
            torch.lgamma(B1_f + 1.0)
            - torch.lgamma(b_minus_clamped.float() + 1.0)
            - torch.lgamma((B1_f - b_minus_clamped.float()) + 1.0)
        )
        logC2_minus = (
            torch.lgamma((N_f - B1_f) + 1.0)
            - torch.lgamma((N_t.float() - b_minus_clamped.float()) + 1.0)
            - torch.lgamma(((N_f - B1_f) - (N_t.float() - b_minus_clamped.float())) + 1.0)
        )
        log_h_minus = logC1_minus + logC2_minus  # (B,d)

        # 4d) compute π₊ = exp(log_h_plus) / [exp(log_h_plus) + exp(log_h_minus)]
        #     do in a stable way:
        max_log = torch.maximum(log_h_plus, log_h_minus)
        h_plus_st = torch.exp(log_h_plus - max_log)
        h_minus_st = torch.exp(log_h_minus - max_log)
        pi_plus = h_plus_st / (h_plus_st + h_minus_st + 1e-16)  # (B,d)
        # clamp to [0,1]
        pi_plus = pi_plus.clamp(0.0, 1.0)

        # 4e) sample a Bernoulli for each coordinate to choose ± sign
        #     pos_mask=True => choose B_t = b_plus; else B_t=b_minus
        rand_unif = torch.rand_like(pi_plus)
        pos_mask = (rand_unif < pi_plus)  # (B,d), boolean

        # 4f) recover the chosen B_t
        B_t_choice = torch.where(pos_mask, b_plus, b_minus)  # (B,d)

        # 4g) compute N_s = floor(N * w(s)), w(s) = Lam[k-1]/Lambda1
        w_s = (Lam[k-1] / Lambda1).clamp(0.0, 1.0)
        N_s = torch.floor(N.float() * w_s).long()  # (B,d)

        # 4h) draw B_s ~ Hypergeom(N_t, B_t_choice, N_s) 
        #     (i.e. among the N_t draws, exactly B_t_choice "successes",
        #     then we draw N_s draws from those N_t without replacement)
        B_s_np = manual_hypergeometric(
            total_count=N_t.cpu().numpy(),
            num_successes=B_t_choice.cpu().numpy(),
            num_draws=N_s.cpu().numpy()
        )
        B_s = torch.from_numpy(B_s_np).to(device)

        # 4i) compute signed X_s = X_t_sign - 2*(B_t_choice - B_s) - (N_t - N_s)
        #     where X_t_sign = (+x_t) if pos_mask else (-x_t)
        sign_t = torch.where(pos_mask, +torch.ones_like(x), -torch.ones_like(x))  # (B,d)
        X_t_signed = sign_t * x  # (B,d)
        X_s_signed = X_t_signed - 2 * (B_t_choice - B_s) - (N_t - N_s)  # (B,d)

        # 4j) reflect to get non-negative x at time s
        x = X_s_signed.abs().long()  # (B,d)

        # 4k) update B1 = B_s for the next iteration
        B1 = B_s

        # optionally record trajectory
        if return_trajectory:
            trajectory.append(x.clone().cpu())

        # Note:  x0_hat remains unchanged (we do not re‐predict mid-trajectory).
        #        We use the same predicted "a" = x0_hat throughout.
        #        If you wish to re-predict at each step, uncomment:
        #        if k > 1:
        #            t_new = torch.full((B,1), t_grid[k-1], device=device)
        #            x0_hat = model.sample(x.float(), z, t_new, use_mean=use_mean).long()

    # End of for‐loop: x is now at t=0
    if return_trajectory:
        return x.long(), trajectory
    return x.long() 