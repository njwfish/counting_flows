import torch
from torch.distributions import Binomial
import numpy as np
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from ...sampling.hypergeom import hypergeometric
from ...sampling.mean_constrained import mh_mean_constrained_update
from ...sampling.distribute_shift import get_proportional_weighted_dist, sample_pert


class SkellamMeanConstrainedBridge:
    """
    Same latent construction as SkellamBridge but the intermediate
    state x_t is *resampled* (via a fast MH swap) so that
            sum_i x_t^{(i)}  =  d * [ (1–t)·mean(x0) + t·mean(x1) ] .
    """

    def __init__(
        self,
        n_steps: int,
        lam0: float = 8.0,
        lam1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        mh_sweeps: int = 5,
        **schedule_kwargs,
    ):
        self.n_steps        = n_steps
        self.lam0, self.lam1 = lam0, lam1
        self.schedule_type  = schedule_type
        self.homogeneous_time = homogeneous_time
        self.mh_sweeps      = mh_sweeps
        self.sched_kwargs   = schedule_kwargs
        self.time_points    = make_time_spacing_schedule(
            n_steps, time_schedule, **schedule_kwargs
        )
        self.lam_p, self.lam_m, self.Λp, self.Λm = make_lambda_schedule(
                self.time_points, self.lam0, self.lam1, self.schedule_type
        )
        self.Λ_tot1 = self.Λp[-1] + self.Λm[-1]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sample_N_M_B1(diff, Λp1, Λm1, device):
        λ = 2.0 * torch.sqrt(Λp1 * Λm1)        # scalar
        λ = λ.unsqueeze(-1).expand_as(diff)   # (B,d)
        M  = torch.poisson(λ).long().to(device)
        N  = diff.abs() + 2 * M
        B1 = (N + diff) // 2.0
        return N, M, B1

    # ------------------------------------------------------------------ main
    def __call__(self, batch, t_target=None):
        # unpack
        if isinstance(batch, list):
            x0 = torch.stack([b["x0"] for b in batch])  # (B,d)
            x1 = torch.stack([b["x1"] for b in batch])
        else:
            x0 = batch["x0"]
            x1 = batch["x1"]

        # dataloader might add an extra dim
        if x0.ndim == 3:
            x0, x1 = x0.squeeze(0), x1.squeeze(0)

        device = x0.device
        B, d   = x0.shape
        self.time_points = self.time_points.to(device)
        self.Λp = self.Λp.to(device)
        self.Λm = self.Λm.to(device)
        self.Λ_tot1 = self.Λ_tot1.to(device)

        # λ–schedule
        Λ_tot1 = self.Λ_tot1

        # latent (N, M, B1)
        diff = x1 - x0
        N, M, B1 = self._sample_N_M_B1(diff, self.Λp[-1], self.Λm[-1], device)

        # choose time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = torch.abs(self.time_points - t_target)
            k_idx_val = torch.argmin(time_diffs).item()
            k_idx = torch.full((B,), k_idx_val, device=device)
            t_val = self.time_points[k_idx_val]
        elif self.homogeneous_time:
            k_idx_val = torch.randint(0, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_val, device=device)
            t_val = self.time_points[k_idx_val]
        else:
            raise NotImplementedError("Non-homogeneous time is not implemented for mean-constrained bridge")

        t      = self.time_points[k_idx]                            # (B,)
        w_t    = (self.Λp[k_idx] + self.Λm[k_idx]) / Λ_tot1       # (B,)
        N_t    = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()  # (B,d)

        # births up to t
        B_t = hypergeometric(N, B1, N_t)

        # ------------------------------------------------------------------
        #  Mean–constraint target  S_target
        # ------------------------------------------------------------------
        mu0  = x0.float().mean(dim=0)                  # (d,)
        mu1  = x1.float().mean(dim=0)                  # (d,)
        mu_t = (1. - t_val) * mu0 + t_val * mu1        # (d,)
        S_target = ((mu_t - mu0) * B).round().long() + N_t.sum(dim=0) // 2

        # MH projection
        B_t = mh_mean_constrained_update(
            N_s    = N_t.clone(),              # same time-slice
            B_s    = B_t.clone(),
            N_t    = N.clone(),
            B_t    = B1.clone(),
            S      = S_target,
            sweeps = self.mh_sweeps,
            backend='torch'
        )
        x_t = x1 - 2 * (B1 - B_t) + (N - N_t)

        # recompute diff & slack pairs after MH (keeps N_t fixed)
        diff_t = (x_t - x0)
        M_t    = (N_t - diff_t.abs()) // 2.0

        return {
            "x0"   : x0,
            "x1"   : x1,
            "x_t"  : x_t,
            "t"    : t,          
            "M_t"    : M_t
        }

    @torch.no_grad()
    def reverse_sampler(
        self,
        x1: torch.LongTensor,
        z:  torch.Tensor,
        mu0: torch.Tensor,
        model,
        device: str = "cuda",
        return_trajectory: bool = False,
        return_x_hat:     bool = False,
        return_M:         bool = False,
        guidance_x0:      torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
    ):
        """
        Reverse sampler that *enforces the running batch mean*
            μ_k = (1−t_k)·μ₀ + t_k·μ₁
        at every grid point via an MH swap kernel.

        Identical to the plain sampler except that, after the BD step,
        the state is projected onto the hyper-plane
            { ‖x‖₁  =  round(B · μ_k) }.
        """
        if not isinstance(z, dict):
            raise ValueError("z must be a dictionary")
        device = torch.device(device)
        x_t    = x1.to(device).long()           # current state (starts at X₁)
        Bbatch, d = x_t.shape

        # ── forward schedule and w(t) ────────────────────────────────────
        w = (self.Λp + self.Λm) / self.Λ_tot1       # (K+1,)

        # ── initial slack-pair count at t=1 ─────────────────────────────
        λ_star = 2. * torch.sqrt(self.Λp[-1] * self.Λm[-1]).expand_as(x_t).to(device)
        M_t    = torch.poisson(λ_star).long()     # (B,d)

        # ── first model call (t=1) ──────────────────────────────────────
        t1         = torch.ones_like(x_t[:, :1])
        x0_hat_t_out   = model.sample(x_t, M_t, t1, **z).round().long()
        
        if guidance_x0 is not None:
            guidance_mix = 1
            x0_hat_t_out = (guidance_mix * guidance_x0 + (1 - guidance_mix) * x0_hat_t_out).round().long()

        weighted_dist = get_proportional_weighted_dist(x0_hat_t_out.float())
        x0_hat_t = sample_pert(x0_hat_t_out, weighted_dist, mu0 - x0_hat_t_out.float().mean(dim=0))

        diff       = x_t - x0_hat_t
        N_t        = diff.abs() + 2 * M_t
        B_t        = (N_t + diff) >> 1

        # target means
        mu1 = x1.float().mean(dim=0) 

        # trackers
        traj, xhat_traj, M_traj = [], [], []

        # ── reverse sweep  t_k → t_{k-1} ────────────────────────────────
        for k in range(self.n_steps, 0, -1):
            ρ = (w[k-1] / w[k]).item()          # thinning prob

            # single model prediction at t_k
            t_tensor  = torch.zeros_like(x_t[:, :1]) + self.time_points[k-1]
            x0_hat_t_out  = model.sample(x_t, M_t, t_tensor, **z).round().long()
            
            if guidance_x0 is not None:
                guidance_mix = guidance_schedule[k-1]
                x0_hat_t_out = (guidance_mix * guidance_x0 + (1 - guidance_mix) * x0_hat_t_out).round().long()

            weighted_dist = get_proportional_weighted_dist(x0_hat_t_out.float())
            x0_hat_t = sample_pert(x0_hat_t_out, weighted_dist, mu0 - x0_hat_t_out.float().mean(dim=0))

            diff      = x_t - x0_hat_t
            N_t       = diff.abs() + 2 * M_t
            B_t       = (N_t + diff) >> 1

            # thin total jumps
            non_zero = N_t > 0
            N_s = torch.zeros_like(N_t)
            N_s[non_zero] = Binomial(N_t[non_zero].float(), ρ).sample().long()

            # births that survive the thinning  ~ Hypergeom
            B_s = torch.zeros_like(B_t)
            B_s[non_zero] = hypergeometric(
                total_count   = N_t[non_zero],
                success_count = B_t[non_zero],
                num_draws     = N_s[non_zero]
            )

            # Metropolis–Hastings mean-projection
            t_s   = self.time_points[k-1].item()
            mu_s  = (1. - t_s) * mu0 + t_s * mu1          # (d,)
            S_k   = ((mu_s - (x0_hat_t.float().mean(dim=0))) * Bbatch).round().long() + N_s.sum(dim=0) // 2       # desired row-sum per sample

            B_s = mh_mean_constrained_update(
                N_s    = N_s.clone(),
                B_s    = B_s.clone(),
                N_t    = N_t.clone(),
                B_t    = B_t.clone(),
                S      = S_k,
                sweeps = self.mh_sweeps,
                override_support=(k == 1),
                backend='torch'
            )
            x_s = x0_hat_t + 2 * B_s - N_s

            # derive new slack pairs AFTER MH projection
            diff_s = (x_s - x0_hat_t).abs()
            M_s    = ((N_s - diff_s) >> 1)                # = min{B_s, D_s}

            # roll forward
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