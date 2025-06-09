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
    
    # Handle reflected BD bridge with dedicated sampler
    if mode == "reflected_bd":
        return reverse_sampler_reflected_bd(
            x1, z, model,
            n_steps=K,
            lam0=lam0,
            lam1=lam1,
            schedule_type=bd_schedule_type,
            use_mean=use_mean,
            device=device,
            return_trajectory=return_trajectory,
            return_x_hat=return_x_hat
        )
    
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


import torch
from torch.distributions import Binomial, NegativeBinomial
from typing import Optional, Tuple, List

@torch.no_grad()
def reverse_sampler_reflected_bd(
    x1: torch.Tensor,
    z: torch.Tensor,
    model,
    *,
    n_steps: int = 50,
    lam0: float = 8.0,
    lam1: float = 8.0,
    schedule_type: str = "constant",
    use_mean: bool = False,
    device: str = "cuda",
    return_trajectory: bool = False,
    return_x_hat: bool = False,
) -> Tuple[torch.LongTensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Reverse‐time sampler for the *reflected* birth–death bridge (equal rates).

    Args:
      x1:             Tensor of shape (B, d), the final state X₁ (absolute value).
      z:              Context tensor, passed to model.sample.
      model:          Trained model with `model.sample(x, z, t, use_mean)`.
      n_steps:        Number of reverse steps (K).
      lam0, lam1:     Birth/death rates at t=0 and t=1.
      schedule_type:  How λ varies over time.
      use_mean:       If True, use model's mean prediction; else sample.
      device:         "cuda" or "cpu".
      return_trajectory: whether to collect the |X_t| trajectory.
      return_x_hat:     whether to also collect the x0_hat predictions.

    Returns:
      x0:             LongTensor (B,d) of sampled X₀ (absolute value).
      trajectory:     List of |X_t| for each reverse step (if requested).
      x_hat_traj:     List of x0_hat for each step (if requested).
    """

    # Move to device and ensure float
    x = x1.clone().to(device).float()          # current absolute state |X_t|
    B, d = x.shape

    # 1) Build the forward‐time rate schedules, then reverse them
    grid, lam_p, lam_m, Lambda_p, Lambda_m = make_lambda_schedule(
        n_steps, lam0, lam1, lam0, lam1, schedule_type, device=device
    )
    # Total integral at t=1
    Lambda_total = Lambda_p[-1] + Lambda_m[-1]

    # Reverse all of them so index 0 corresponds to t=1, index K to t=0
    grid_rev     = torch.flip(grid,     dims=[0]).to(device)  # length K+1
    Lambda_p_rev = torch.flip(Lambda_p, dims=[0]).to(device)
    Lambda_m_rev = torch.flip(Lambda_m, dims=[0]).to(device)

    # Precompute the (K+1)-length vector of cumulative‐jump probabilities
    ratios = (Lambda_p_rev + Lambda_m_rev) / Lambda_total  # shape (K+1,)

    # 2) Sample the *static* latent total jumps N and B₁ at t=1
    #    Predict x0_hat at t=1
    t1       = torch.ones((B, 1), device=device)
    x0_hat_1 = model.sample(x, z, t1, use_mean=use_mean).float()  # shape (B,d)

    #    Integerize the residual diff = X₁ - a (a = x0_hat_1)
    diff0 = (x - x0_hat_1).round().long()  # signed count difference
    #    Sample Poisson M ~ Poi(2√(Λp(1)Λm(1))) and form N, B1
    λstar = 2.0 * torch.sqrt(Lambda_p_rev[0] * Lambda_m_rev[0])            # scalar
    λstar = λstar.unsqueeze(-1).expand_as(diff0)                           # (B,d)
    M     = torch.poisson(λstar).long()                                    # (B,d)
    N     = diff0.abs() + 2*M                                              # total jumps by t=1
    B1    = ((N + diff0) >> 1)                                             # births by t=1

    # 3) Initialize the latent *sign* S₁ from diff0
    sign = torch.sign(diff0).float()                                      # ±1 or 0
    zeros = (sign == 0).float()
    if zeros.any():
        # break ties at zero randomly
        rnd = torch.rand_like(sign[zeros], device=device).float()
        sign[zeros] = torch.where(rnd < 0.5, 1.0, -1.0)

    # 4) Optionally store trajectories
    if return_trajectory:
        traj = [x.clone().cpu()]
    if return_trajectory and return_x_hat:
        xh_traj = [x0_hat_1.clone().cpu()]

    # 5) Initialize the first cumulative‐jump count N_t = N_{t₁} with a Binomial
    p0 = ratios[0].unsqueeze(-1).expand_as(N)   # probability at t=1
    N_t = Binomial(total_count=N.float(), probs=p0).sample().long()

    # 6) Reverse‐time steps
    for k in range(n_steps):
        # current / next jump‐fraction
        w_cur = ratios[k]
        w_nxt = ratios[k+1]

        # draw N_{t_{k+1}} from N_{t_k}
        p_ratio = (w_nxt / w_cur).clamp(max=1.0)
        p_ratio = p_ratio.unsqueeze(-1).expand_as(N_t)
        N_s = Binomial(total_count=N_t.float(), probs=p_ratio).sample().long()

        # predict new baseline a = x0_hat at current t
        t_cur    = grid_rev[k]
        t_tensor = torch.full((B,1), t_cur, device=device)
        x0_hat   = model.sample(x, z, t_tensor, use_mean=use_mean).float()

        # store x0_hat if desired
        if return_trajectory and return_x_hat:
            xh_traj.append(x0_hat.clone().cpu())

        # rebuild signed X_t from (sign × |X_t|)
        X_signed = sign * x

        # snap residual δ = X_signed – a
        delta = (X_signed - x0_hat).round().long()
        # clamp to feasible range and enforce parity
        delta = torch.clamp(delta, -N_t, N_t)
        delta = delta - ((delta + N_t) & 1)

        # two candidate birth counts that yield |X_t|=x
        #   B_t⁺ = (N_t + ( +x - a)) / 2
        #   B_t⁻ = (N_t + ( -x - a)) / 2
        xp = x.round().long()
        ap = x0_hat.round().long()
        B_pos = ((N_t + ( xp - ap)) >> 1).clamp(0, N_t)
        B_neg = ((N_t + (-xp - ap)) >> 1).clamp(0, N_t)

        # compute log‐weights ∝ Hypergeom pmf via log‐combinatorics
        #  log C(B1, B) + log C(N-B1, N_t-B)
        def log_hyp(N, K, n, k):
            # log C(K,k) + log C(N-K, n-k) - log C(N,n):
            return (
                torch.lgamma(K + 1) - torch.lgamma(k + 1) - torch.lgamma(K - k + 1)
                + torch.lgamma(N - K + 1)
                  - torch.lgamma(n - k + 1)
                  - torch.lgamma((N - K) - (n - k) + 1)
            )
        logw_pos = log_hyp(N, B1, N_t, B_pos)
        logw_neg = log_hyp(N, B1, N_t, B_neg)
        pi_plus  = torch.sigmoid(logw_pos - logw_neg)

        # sample new sign
        u_new = torch.rand_like(pi_plus)
        sign  = torch.where(u_new < pi_plus, 1, -1)

        # pick the branch's birth count
        B_t = torch.where(sign > 0, B_pos, B_neg)

        # hypergeometric thinning: sample births in [0,s]
        B_s_np = manual_hypergeometric(
            total_count=N_t.cpu().numpy(),
            num_successes=B_t.cpu().numpy(),
            num_draws=N_s.cpu().numpy()
        )
        B_s = torch.from_numpy(B_s_np).to(device).long()

        # update signed state by removing the s→t jumps
        X_signed = X_signed - 2 * (B_t - B_s) + (N_t - N_s)
        # observe new absolute value
        x = X_signed.abs()

        # prepare for next iteration
        N_t = N_s

        # store trajectory
        if return_trajectory:
            traj.append(x.clone().cpu())

    # cast to integer output
    x0 = x.long()

    if return_trajectory:
        if return_x_hat:
            return x0, traj, xh_traj
        else:
            return x0, traj
    else:
        return x0
