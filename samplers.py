"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial
from .scheduling import make_lambda_schedule
from .bridges import manual_hypergeometric, mh_mean_constrained_update


@torch.no_grad()
def bd_reverse_sampler(
    x1:  torch.LongTensor,          # (B,d) observed X₁
    z:   torch.Tensor,              # conditioning / context
    model,                          # nn: (x_t,M_t,z,t) ↦ x̂₀
    K:         int,                 # # grid steps
    lam0:      float,               # λ₊(0)=λ₋(0)
    lam1:      float,               # λ₊(1)=λ₋(1)
    schedule_type:  str = "constant",
    time_schedule:  str = "uniform",
    return_trajectory: bool = False,
    return_x_hat:     bool = False,
    return_M:         bool = False,
    device: str = "cuda",
    x0 = None,
):
    """
    Exact reverse sampler for the *standard* birth–death bridge.
    The slack-pair counter M_t is **never thinned directly**; it is
    recomputed from (N_s,B_s) at every step, which preserves the
    correct joint law.
    """
    device  = torch.device(device)
    x_t     = x1.to(device).long()              # starts at X₁
    B, d    = x_t.shape

    # ── forward schedule ────────────────────────────────────────────
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1, schedule_type, device=device
    )
    w  = (Λp + Λm) / (Λp[-1] + Λm[-1])          # w(t_k)

    # ── latent totals at t=1 ────────────────────────────────────────
    λ_star = 2. * torch.sqrt(Λp[-1] * Λm[-1]).expand_as(x_t)   # scalar
    M_t    = torch.poisson(λ_star).long()   # (B,d)

    t1          = torch.ones_like(x_t[:, :1])
    x0_hat_t    = model.sample(x_t, M_t, z, t1).round().long()  # if x0 is None else x0.to(device).long()
    diff        = x_t - x0_hat_t
    N_t         = diff.abs() + 2 * M_t
    B_t         = (N_t + diff) >> 1             # births at t₁

    # optional logs
    traj, xhat_traj, M_traj = [], [], []

    # ── reverse sweep:  t_k → t_{k-1} ───────────────────────────────
    for k in range(K, 0, -1):
        ρ = (w[k-1] / w[k]).item()              # thinning prob

        # a) model prediction of x̂₀ at current time t_k
        t_tensor  = torch.zeros_like(x_t[:, :1]) + grid[k-1]
        x0_hat_t    = model.sample(x_t, M_t, z, t_tensor).round().long() # if x0 is None else x0.to(device).long()

        # b) update latent (N_t, B_t) with *current* M_t
        diff  = x_t - x0_hat_t
        N_t   = diff.abs() + 2 * M_t
        B_t   = (N_t + diff) >> 1

        # c) thin the TOTAL jump count
        N_s = Binomial(N_t.float(), ρ).sample().long()

        # d) births surviving the thinning  ~ Hypergeom
        B_s = manual_hypergeometric(
            total_count   = N_t,
            success_count = B_t,
            num_draws     = N_s
        )
        # B_s = torch.from_numpy(B_s_np).to(device).long()

        # e) reconstruct state at s and *derive* slack pairs
        x_s = x0_hat_t + 2*B_s - N_s
        diff_s = (x_s - x0_hat_t).abs()
        M_s = ((N_s - diff_s) >> 1)   

        # f) roll forward
        x_t, M_t = x_s, M_s

        # logs
        if return_trajectory: traj.append(x_t.cpu())
        if return_x_hat:     xhat_traj.append(x0_hat_t.cpu())
        if return_M:         M_traj.append(M_t.cpu())

    # ── return ──────────────────────────────────────────────────────
    outs = [x_t]
    if return_trajectory: outs.append(traj)
    if return_x_hat:     outs.append(xhat_traj)
    if return_M:         outs.append(M_traj)
    return tuple(outs) if len(outs) > 1 else x_t


@torch.no_grad()
def bd_reverse_with_interpolated_mean(
    x1: torch.LongTensor,                   # (B,d) terminal counts
    z:  torch.Tensor,                       # context
    model,                                  # NN: sample(x_t, M_t, z, t)
    K:        int,
    lam0:     float, lam1: float,           # λ₊(0)=λ₋(0)=lam0 … λ₊(1)=λ₋(1)=lam1
    mu0,                                    # scalar or (d,) desired mean at t=0
    sweeps:   int = 5,                      # MH sweeps / step
    schedule_type: str = "constant",
    time_schedule: str = "uniform",
    device: str = "cuda",
    return_trajectory: bool = False,
    return_x_hat:     bool = False,
    return_M:         bool = False,
):
    """
    Reverse sampler that *enforces the running batch mean*
        μ_k = (1−t_k)·μ₀ + t_k·μ₁
    at every grid point via an MH swap kernel.

    Identical to the plain sampler except that, after the BD step,
    the state is projected onto the hyper-plane
        { ‖x‖₁  =  round(B · μ_k) }.
    """
    device = torch.device(device)
    x_t    = x1.to(device).long()           # current state (starts at X₁)
    Bbatch, d = x_t.shape

    # ── forward schedule and w(t) ────────────────────────────────────
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1,
        schedule_type, device=device
    )
    w = (Λp + Λm) / (Λp[-1] + Λm[-1])       # (K+1,)

    # ── initial slack-pair count at t=1 ─────────────────────────────
    λ_star = 2. * torch.sqrt(Λp[-1] * Λm[-1]).expand_as(x_t)
    M_t    = torch.poisson(λ_star).long()     # (B,d)

    # ── first model call (t=1) ──────────────────────────────────────
    t1         = torch.ones_like(x_t[:, :1])
    x0_hat_t   = model.sample(x_t, M_t, z, t1).round().long()
    diff       = x_t - x0_hat_t
    N_t        = diff.abs() + 2 * M_t
    B_t        = (N_t + diff) >> 1

    # target means
    mu1 = x1.float().mean(dim=0)            # (d,)
    mu0 = torch.as_tensor(mu0, device=device)
    if mu0.ndim == 0:                       # scalar → broadcast
        mu0 = mu0.repeat(d)

    # trackers
    traj, xhat_traj, M_traj = [], [], []

    # ── reverse sweep  t_k → t_{k-1} ────────────────────────────────
    for k in range(K, 0, -1):
        ρ = (w[k-1] / w[k]).item()          # thinning prob

        # 1) single model prediction at t_k
        t_tensor  = torch.zeros_like(x_t[:, :1]) + grid[k-1]
        x0_hat_t  = model.sample(x_t, M_t, z, t_tensor).round().long()
        diff      = x_t - x0_hat_t
        N_t       = diff.abs() + 2 * M_t
        B_t       = (N_t + diff) >> 1

        print("Min N_t: ", N_t.min(), "Max N_t: ", N_t.max(), "ρ: ", ρ)
        print("Min diff: ", diff.abs().min(), "Max diff: ", diff.abs().max())
        print("M_t: ", M_t.min(), "Max M_t: ", M_t.max())

        # 2) thin total jumps
        N_s = Binomial(N_t.float(), ρ).sample().long()

        # 3) births that survive the thinning  ~ Hypergeom
        B_s = manual_hypergeometric(
            total_count   = N_t, #.cpu().numpy(),
            success_count = B_t, #.cpu().numpy(),
            num_draws     = N_s #.cpu().numpy()
        )
        # B_s = torch.from_numpy(B_s_np).to(device).long()

        # 4) raw state at s
        x_raw_s = x0_hat_t + 2*B_s - N_s

        # 5) Metropolis–Hastings mean-projection
        t_s   = grid[k-1].item()
        mu_s  = (1. - t_s) * mu0 + t_s * mu1          # (d,)
        S_k   = (mu_s * Bbatch).round().long()        # desired row-sum per sample

        # expected counts for proposal weighting
        E_Bs  = B_t.float().clamp(min=1e-6) / N_t.float().clamp(min=1e-6) * N_s.float()
        x_exp = x0_hat_t.float() + 2*E_Bs - N_s.float()

        x_s = mh_mean_constrained_update(
            x      = x_raw_s,
            x_exp  = x_exp,
            S      = S_k,
            a      = x0_hat_t.clone(),
            N_t    = N_t.clone(),
            B_t    = B_t.clone(),
            N_s    = N_s.clone(),
            sweeps = sweeps
        )

        # 6) derive new slack pairs AFTER MH projection
        diff_s = (x_s - x0_hat_t).abs()
        M_s    = ((N_s - diff_s) >> 1)                # = min{B_s, D_s}

        # 7) roll forward
        x_t, M_t = x_s, M_s
        N_t, B_t = N_s, ((N_s + (x_s - x0_hat_t)) >> 1)

        if return_trajectory: traj.append(x_s.cpu())
        if return_x_hat:      xhat_traj.append(x0_hat_t.cpu())
        if return_M:          M_traj.append(M_s.cpu())

    # ── output ──────────────────────────────────────────────────────
    outs = [x_t]
    if return_trajectory: outs.append(traj)
    if return_x_hat:      outs.append(xhat_traj)
    if return_M:          outs.append(M_traj)
    return tuple(outs) if len(outs) > 1 else x_t


def reflected_bd_reverse_sampler(
    x1, z, model,
    K,
    lam0, lam1,
    schedule_type="constant",
    time_schedule="uniform",
    device="cuda",
    return_trajectory=False,
    return_x_hat=False,
    return_M=False,
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
        return_M=return_M,
        device=device
    )