import torch
from torch.distributions import Binomial
import numpy as np
from ..scheduling import make_time_spacing_schedule, make_lambda_schedule
from ..sampling.hypergeom import hypergeometric
from ..sampling.mean_constrained import mh_mean_constrained_update



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
        B1 = (N + diff) >> 1
        return N, M, B1

    # ------------------------------------------------------------------ main
    def __call__(self, batch, t_target=None):
        # unpack
        x0 = torch.stack([b["x0"] for b in batch])  # (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])

        # dataloader might add an extra dim
        if x0.ndim == 3:
            x0, x1, z = x0.squeeze(0), x1.squeeze(0), z.squeeze(0)

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

        # raw x_t
        x_t_raw = x0 + 2 * B_t - N_t                   # (B,d)

        # ------------------------------------------------------------------
        #  Mean–constraint target  S_target
        # ------------------------------------------------------------------
        mu0  = x0.float().mean(dim=0)                  # (d,)
        mu1  = x1.float().mean(dim=0)                  # (d,)
        mu_t = (1. - t_val) * mu0 + t_val * mu1        # (d,)
        S_target = (mu_t * B).round().long()           # (B,)

        # expected x for proposal weights
        p_birth = B1.float() / N.float()
        E_Bt    = p_birth * N_t.float()
        x_exp   = x0.float() + 2*E_Bt - N_t.float()

        # MH projection
        x_t = mh_mean_constrained_update(
            x      = x_t_raw,
            x_exp  = x_exp,
            S      = S_target,
            a      = x0.clone(),               # here a = X₀, not x̂₀
            N_t    = N_t.clone(),
            B_t    = B_t.clone(),
            N_s    = N_t.clone(),              # same time-slice
            sweeps = self.mh_sweeps
        )

        # recompute diff & slack pairs after MH (keeps N_t fixed)
        diff_t = (x_t - x0)
        M_t    = ((N_t - diff_t.abs()) >> 1)

        return {
            "x0"   : x0,
            "x1"   : x1,
            "x_t"  : x_t,
            "t"    : t,          
            "z"    : z,
            "M"    : M_t
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
        w = (self.Λp + self.Λm) / self.Λ_tot1       # (K+1,)

        # ── initial slack-pair count at t=1 ─────────────────────────────
        λ_star = 2. * torch.sqrt(self.Λp[-1] * self.Λm[-1]).expand_as(x_t).to(device)
        M_t    = torch.poisson(λ_star).long()     # (B,d)

        # ── first model call (t=1) ──────────────────────────────────────
        t1         = torch.ones_like(x_t[:, :1])
        x0_hat_t   = model.sample(x_t, M_t, z, t1).round().long()
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
            x0_hat_t  = model.sample(x_t, M_t, z, t_tensor).round().long()
            diff      = x_t - x0_hat_t
            N_t       = diff.abs() + 2 * M_t
            B_t       = (N_t + diff) >> 1

            # thin total jumps
            N_s = Binomial(N_t.float(), ρ).sample().long()

            # births that survive the thinning  ~ Hypergeom
            B_s = hypergeometric(
                total_count   = N_t,
                success_count = B_t,
                num_draws     = N_s
            )

            # raw state at s
            x_raw_s = x0_hat_t + 2*B_s - N_s

            # Metropolis–Hastings mean-projection
            t_s   = self.time_points[k-1].item()
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
                sweeps = self.mh_sweeps,
                override_support=(k == 1)
            )

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