import torch
from torch.distributions import Binomial
import numpy as np
from ..bridges.scheduling import make_time_spacing_schedule, make_lambda_schedule
from ..sampling.hypergeom import hypergeometric



class SkellamBridge:
    """
    Collate for Skellam birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    """

    def __init__(
        self,
        n_steps: int,
        lam0: float = 8.0,
        lam1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam0 = lam0
        self.lam1 = lam1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.sched_kwargs = schedule_kwargs
        
        # Create time spacing
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

        self.lam_p, self.lam_m, self.Λp, self.Λm = make_lambda_schedule(
            self.time_points, self.lam0, self.lam1, schedule_type
        )
        self.Λ_tot1 = self.Λp[-1] + self.Λm[-1]


    def _sample_N_B1(self, diff: torch.Tensor, Lambda_p: torch.Tensor, Lambda_m: torch.Tensor, device):
        """
        diff = x1 - x0   (shape [B, d])
        Lambda_p, Lambda_m: Final cumulative integrals from lambda schedule (scalars)
        Returns N (total jumps) and B1 (birth count at t=1).
        Uses Poisson for total jumps with lambda_star from schedule.
        """
        B, d = diff.shape
        # Use the cumulative lambda integrals
        lambda_star = 2.0 * torch.sqrt(Lambda_p * Lambda_m).expand_as(diff)        # scalar
        M = torch.poisson(lambda_star).long().to(device)
        N = diff.abs() + 2 * M
        B1 = (N + diff) // 2   # integer division
        return N, M, B1

    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx
        return t, t_idx

    def __call__(self, batch, t_target=None):
        # batch = list of dicts with keys 'x0','x1','z'
        x0 = torch.stack([b["x0"] for b in batch])  # (1,B,d) -> squeeze to (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
        
        # Handle case where dataset already produces batches (extra dimension from DataLoader)
        if len(x0.shape) == 3:  # (1, B, d)
            x0 = x0.squeeze(0)  # (B, d)
            x1 = x1.squeeze(0)  # (B, d)
            z = z.squeeze(0)    # (B, context_dim)
        
        device = x0.device
        B, d = x0.shape
        self.time_points = self.time_points.to(device)
        self.Λp = self.Λp.to(device)
        self.Λm = self.Λm.to(device)
        self.Λ_tot1 = self.Λ_tot1.to(device)

        # latent (N,B1)
        diff = (x1 - x0)          # (B, d) - don't squeeze
        N, M, B1 = self._sample_N_B1(diff, self.Λp[-1], self.Λm[-1], device)       # shape (B, d)

        # pick time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = torch.abs(self.time_points - t_target)
            k_idx_scalar = torch.argmin(time_diffs).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        elif self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx_scalar = torch.randint(0, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        else:
            # Sample different time indices for each sample (original behavior)
            k_idx = torch.randint(1, self.n_steps + 1, (B,), device=device)
            
            
        t     = self.time_points[k_idx]                               # (B,)
        w_t   = (self.Λp[k_idx] + self.Λm[k_idx]) / self.Λ_tot1          # (B,)
        N_t   = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()       # (B, d)

        # draw B_t ~ Hypergeom(N, B1, N_t)
        B_t = hypergeometric(
            total_count=N,
            success_count=B1,
            num_draws=N_t
        )

        x_t = x0 + (2 * B_t - N_t)          # (B, d)

        diff_t = (x_t - x0).abs()          # (B,d)
        M_t    = ((N_t - diff_t) >> 1)     # integer tensor (B,d)

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "M" : M_t, 
        }
    
    @torch.no_grad()
    def reverse_sampler(
        self,
        x1:  torch.LongTensor,          # (B,d) observed X₁
        z:   torch.Tensor,              # conditioning / context
        model,                          # nn: (x_t,M_t,z,t) ↦ x̂₀
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
        w  = (self.Λp + self.Λm) / self.Λ_tot1          # w(t_k)

        # ── latent totals at t=1 ────────────────────────────────────────
        λ_star = 2. * torch.sqrt(self.Λp[-1] * self.Λm[-1]).expand_as(x_t).to(device)   # scalar
        M_t    = torch.poisson(λ_star).long()   # (B,d)

        t1          = torch.ones_like(x_t[:, :1])
        x0_hat_t    = model.sample(x_t, M_t, z, t1).round().long()  # if x0 is None else x0.to(device).long()
        diff        = x_t - x0_hat_t
        N_t         = diff.abs() + 2 * M_t
        B_t         = (N_t + diff) >> 1             # births at t₁

        # optional logs
        traj, xhat_traj, M_traj = [], [], []

        # ── reverse sweep:  t_k → t_{k-1} ───────────────────────────────
        for k in range(self.n_steps, 0, -1):
            ρ = (w[k-1] / w[k]).item()              # thinning prob

            # model prediction of x̂₀ at current time t_k
            t_tensor  = torch.zeros_like(x_t[:, :1]) + self.time_points[k-1]
            x0_hat_t    = model.sample(x_t, M_t, z, t_tensor).round().long() # if x0 is None else x0.to(device).long()

            # update latent (N_t, B_t) with *current* M_t
            diff  = x_t - x0_hat_t
            N_t   = diff.abs() + 2 * M_t
            B_t   = (N_t + diff) >> 1

            # thin the TOTAL jump count
            N_s = Binomial(N_t.float(), ρ).sample().long()

            # births surviving the thinning  ~ Hypergeom
            B_s = hypergeometric(
                total_count   = N_t,
                success_count = B_t,
                num_draws     = N_s
            )

            # reconstruct state at s and *derive* slack pairs
            x_s = x0_hat_t + 2 * B_s - N_s
            diff_s = (x_s - x0_hat_t).abs()
            M_s = ((N_s - diff_s) >> 1)   

            # roll forward
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