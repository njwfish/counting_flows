"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial, Beta, NegativeBinomial
from .scheduling import make_time_spacing_schedule, make_phi_schedule, get_bridge_parameters, make_lambda_schedule
from .bridges import manual_hypergeometric, mh_mean_constrained_update

from typing import Tuple

import torch
from torch.distributions import Multinomial



# ------------------------------------------------------------------------------
#  Core BD step (unreflected)
# ------------------------------------------------------------------------------
import torch
from torch.distributions import Multinomial, Binomial

# --------------------------------------------------------------------
#  1) Core BD reverse‐kernel step (unreflected)
# --------------------------------------------------------------------
def bd_step(
    x_t:    torch.LongTensor,   # (B,d) current state at time t
    x0_hat: torch.LongTensor,   # (B,d) model prediction of X₀
    N_t:    torch.LongTensor,   # (B,d) latent total jumps at t
    B_t:    torch.LongTensor,   # (B,d) latent births at t
    w_t:    torch.Tensor,       # scalar or (B,)                = w(t)
    w_s:    torch.Tensor        # scalar or (B,)                = w(s)
) -> (torch.LongTensor, torch.LongTensor, torch.LongTensor):
    """
    One reverse step of the Birth–Death bridge (unreflected).
    Returns (x_s, N_s, B_s).
    """
    # 1) thinning: N_s ~ Binomial(N_t, w_s/w_t)
    p = (w_s / w_t).clamp(0.0, 1.0)
    if p.ndim == 1:
        p = p.unsqueeze(-1)         # (B,1)
    p = p.expand_as(N_t).float()    # (B,d)
    # use torch.distributions.Binomial for true Binomial
    binom = Binomial(total_count=N_t.float(), probs=p)
    N_s   = binom.sample().long()   # (B,d)

    # 2) births: B_s ~ Hypergeom(N_t, B_t, N_s)
    #    (replace with your fast vectorized version)
    B_s_np = manual_hypergeometric(
      total_count=N_t.cpu().numpy(),
      num_successes=B_t.cpu().numpy(),
      num_draws=N_s.cpu().numpy()
    )
    B_s = torch.from_numpy(B_s_np).to(N_t.device).long()

    # 3) reconstruct
    x_s = x0_hat + 2*B_s - N_s
    return x_s, N_s, B_s

# --------------------------------------------------------------------
#  Standard BD reverse‑time sampler WITH slack‑pair feature
# --------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
#  Required imports + helper
# ──────────────────────────────────────────────────────────────────────────────
import torch
from torch.distributions import Binomial

from .bridges import manual_hypergeometric


# ──────────────────────────────────────────────────────────────────────────────
#  Reverse–time sampler for the *standard* Birth–Death bridge
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def bd_reverse_sampler(
    x1:  torch.LongTensor,            # observed end‑point X₁  (B,d)
    z:   torch.Tensor,                # conditioning / context
    model,                            # nn predicting x̂₀(·, t, z, extra)
    K:         int,                   # #grid steps
    lam0:      float,                 # λ₊(0)=λ₋(0)
    lam1:      float,                 # λ₊(1)=λ₋(1)
    schedule_type:  str = "constant",
    time_schedule:  str = "uniform",
    return_trajectory: bool = False,
    return_x_hat:     bool = False,
    device: str = "cuda",
):
    """
    Exact reverse sampler with global slack‑pair counter M, thinned each step.
    At every grid time we pass M_t to the network.
    """
    device  = torch.device(device)
    x_t     = x1.to(device).long()        # current state (starts at X₁)
    B, d    = x_t.shape

    # 0.  Forward rate schedule and w(t)
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1, schedule_type, device=device
    )
    w = (Λp + Λm) / (Λp[-1] + Λm[-1])     # shape (K+1,)

    # 1.  Global slack‑pair count  M_global  (Poisson)
    λ_star   = 2.0 * torch.sqrt(Λp[-1] * Λm[-1])            # scalar tensor
    M_global = torch.poisson(λ_star).long().expand_as(x_t)  # (B,d)
    M_t      = M_global.clone()                             # at t = 1

    # 2.  First network prediction  (t = 1)
    t1        = torch.ones_like(x_t[:, :1])
    x0_hat_t  = model.sample(x_t, M_t, z, t1).round().long()
    diff      = x_t - x0_hat_t
    N_t       = diff.abs() + 2 * M_t                        # Eq. (N = |diff|+2M)
    B_t       = (N_t + diff) >> 1                           # births (integer)

    # optional logging
    traj, xhat_traj = [], []

    # 3.  Reverse sweep  t_k → t_{k-1}
    for k in range(K, 0, -1):
        t_val, s_val = grid[k], grid[k - 1]
        ρ            = (w[k - 1] / w[k]).item()             # scalar (0,1)

        # 3.a  Thin slack pairs:  M_s | M_t ~ Bin(M_t, ρ²)
        M_s = torch.distributions.Binomial(
                  total_count=M_t.float(), probs=ρ**2
        ).sample().long()

        # 3.b  Network prediction at *current* time (t_k)
        t_tensor = torch.full_like(x_t[:, :1], t_val)
        x0_hat_t = model.sample(x_t, M_t, z, t_tensor).round().long()
        diff     = x_t - x0_hat_t

        #      Re‑define N_t, B_t from diff and *current* M_t
        N_t = diff.abs() + 2 * M_t
        B_t = (N_t + diff) >> 1

        # 3.c  Binomial thinning of total jumps:  N_s | N_t
        N_s = torch.distributions.Binomial(
                  total_count=N_t.float(), probs=ρ
        ).sample().long()

        # 3.d  Hypergeometric thinning of births:  B_s | (N_t, B_t, N_s)
        B_s_np = manual_hypergeometric(
            total_count=N_t.cpu().numpy(),
            num_successes=B_t.cpu().numpy(),
            num_draws=N_s.cpu().numpy()
        )
        B_s = torch.from_numpy(B_s_np).to(device).long()


        # 3.f  Roll state for next iteration
        x_t = x_t - 2 * (B_t - B_s) + (N_t - N_s)
        M_t = M_s 

        if return_trajectory:
            traj.append(x_t.cpu())
        if return_x_hat:
            xhat_traj.append(x0_hat_t.cpu())

    # 4.  Return
    if return_trajectory and return_x_hat:
        return x_t, traj, xhat_traj
    if return_trajectory:
        return x_t, traj
    if return_x_hat:
        return x_t, xhat_traj
    return x_t

# ─────────────────────────────────────────────────────────────────────────────
#  Mean-constrained BD reverse sampler (updated to new slack-pair logic)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def bd_reverse_with_interpolated_mean(
    x1: torch.LongTensor,                   # (B,d) terminal counts
    z:  torch.Tensor,                       # context
    model,                                  # NN:  sample(x_t, M_t, z, t)
    K:        int,
    lam0:     float, lam1: float,           # λ₊(0)=λ₋(0)=lam0  … λ₊(1)=λ₋(1)=lam1
    mu0,                                    # (d,) desired mean at t=0
    sweeps:   int = 5,                      # MH sweeps per time step
    schedule_type: str = "constant",
    time_schedule: str = "uniform",
    device: str = "cuda",
    return_trajectory: bool = False,
    return_x_hat: bool = False,
):
    """
    Reverse sampler that **enforces the running batch mean**
        μ_k = (1–t_k) μ₀ + t_k μ₁
    (μ₁ =  X₁-mean) at every grid point, via an MH swap kernel.

    It is identical to the plain sampler except for the
    MH block that projects x_s onto the hyper-plane
    { ∑_i x_s^{(i)} = round(μ_k·d) }.
    """
    device = torch.device(device)
    x_t    = x1.to(device).long()           # current state
    B, d   = x_t.shape

    # ── forward rate schedule & w(t) ────────────────────────────────────────
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1, schedule_type, device=device
    )
    w = (Λp + Λm) / (Λp[-1] + Λm[-1])       # (K+1,)

    # ── global slack-pair count  M  ────────────────────────────────────────
    λ       = 2. * torch.sqrt(Λp[-1] * Λm[-1])
    M_global = torch.poisson(λ).long().expand_as(x_t)   # (B,d)
    M_t      = M_global.clone()                          #  at t=1

    # ── first model prediction (t=1) ───────────────────────────────────────
    t1         = torch.ones_like(x_t[:, :1])
    x0_hat_t   = model.sample(x_t, M_t, z, t1).round().long()
    diff       = x_t - x0_hat_t
    N_t        = diff.abs() + 2 * M_t
    B_t        = (N_t + diff) >> 1        # births

    # mean targets
    mu1   = x1.float().mean(dim=0)        # (d,)

    # optional trackers
    traj, xhat_traj = [], []

    # ── reverse sweep  t_k → t_{k-1} ───────────────────────────────────────
    for k in range(K, 0, -1):
        t_val, s_val = grid[k], grid[k-1]
        ρ            = w[k-1] / w[k]              # tensor

        # 1) thin slack pairs :  M_s | M_t  ~  Bin(M_t, ρ²)
        M_s = Binomial(total_count=M_t.float(), probs=(ρ**2)).sample().long()

        # 2) network prediction at t_k (single call)
        t_tensor  = torch.full_like(x_t[:, :1], t_val)
        x0_hat_t  = model.sample(x_t, M_t, z, t_tensor).round().long()
        diff      = x_t - x0_hat_t
        N_t       = diff.abs() + 2 * M_t
        B_t       = (N_t + diff) >> 1

        # 3) thin total jumps & births
        N_s = Binomial(total_count=N_t.float(), probs=ρ).sample().long()
        B_s = torch.from_numpy(
                 manual_hypergeometric(N_t.cpu().numpy(),
                                       B_t.cpu().numpy(),
                                       N_s.cpu().numpy())
             ).to(device).long()

        # 4) raw reverse update  (no model call at s)
        x_raw_s = x_t - 2*(B_t - B_s) + (N_t - N_s)


        # 5) MH **mean-constraint**  ∑ x = round(μ_s · d)
        μ_s   = (1. - s_val) * mu0 + s_val * mu1     # (d,)
        S_k   = (μ_s * B).round().long()               # (B,)

        #    expected x for proposal weights
        E_Bs  = B_t.float().clamp(min=1e-6) / N_t.float().clamp(min=1e-6) * N_s.float()
        x_exp = x0_hat_t.float() + 2*E_Bs - N_s.float()

        x_s   = mh_mean_constrained_update(
                    x      = x_raw_s,
                    x_exp  = x_exp,
                    S      = S_k,
                    a      = x0_hat_t,
                    N_t    = N_t,
                    B_t    = B_t,
                    N_s    = N_s,
                    sweeps = sweeps
                )

        # 6) roll everything for the next step
        x_t, M_t = x_s, M_s
        N_t, B_t = N_s, ((N_s + (x_s - x0_hat_t)) >> 1)

        if return_trajectory: traj.append(x_s.cpu())
        if return_x_hat:      xhat_traj.append(x0_hat_t.cpu())

    # ── return ─────────────────────────────────────────────────────────────
    if return_trajectory and return_x_hat:
        return x_t, traj, xhat_traj
    if return_trajectory:
        return x_t, traj
    if return_x_hat:
        return x_t, xhat_traj
    return x_t



def reflected_bd_reverse_sampler(
    x1, z, model,
    K,
    lam0, lam1,
    schedule_type="constant",
    time_schedule="uniform",
    device="cuda",
    return_trajectory=False,
    return_x_hat=False,
):
    """
    Reflected BD reverse sampler - delegates to regular BD sampler for now.
    TODO: Implement proper reflected sampling logic.
    """
    # For now, delegate to the regular BD sampler
    return bd_reverse_sampler(
        x1=x1, z=z, model=model,
        K=K, lam0=lam0, lam1=lam1,
        schedule_type=schedule_type,
        time_schedule=time_schedule,
        return_trajectory=return_trajectory,
        return_x_hat=return_x_hat,
        device=device
    )