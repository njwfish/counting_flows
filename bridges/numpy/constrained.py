import numpy as np
import torch
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from ...sampling.hypergeom import hypergeometric_numpy
from ...sampling.mean_constrained import mh_mean_constrained_update
from ...sampling.distribute_shift_numpy import get_proportional_weighted_dist, sample_pert

class SkellamMeanConstrainedBridge:
    """
    Same latent construction as SkellamBridge but the intermediate
    state x_t is *resampled* (via a fast MH swap) so that
            sum_i x_t^{(i)}  =  d * [ (1–t)·mean(x0) + t·mean(x1) ] .
    
    Numpy implementation - identical to PyTorch version.
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
    def _sample_N_M_B1(diff, Λp1, Λm1):
        λ = 2.0 * np.sqrt(Λp1 * Λm1)        # scalar
        λ = np.broadcast_to(λ, diff.shape)   # (B,d)
        M  = np.random.poisson(λ).astype(np.int64)
        N  = np.abs(diff) + 2 * M
        B1 = (N + diff) // 2
        return N, M, B1

    # ------------------------------------------------------------------ main
    def __call__(self, batch, t_target=None):
        # unpack
        if isinstance(batch, list):
            x0 = np.stack([b["x0"] for b in batch])  # (B,d)
            x1 = np.stack([b["x1"] for b in batch])
        else:
            x0 = batch["x0"].astype(np.int64)
            x1 = batch["x1"].astype(np.int64)

        # dataloader might add an extra dim
        if x0.ndim == 3:
            x0, x1 = x0.squeeze(0), x1.squeeze(0)

        # Convert to numpy if needed
        if hasattr(x0, 'cpu'):  # if it's a torch tensor
            x0 = x0.cpu().numpy()
        if hasattr(x1, 'cpu'):  # if it's a torch tensor
            x1 = x1.cpu().numpy()
            
        # Convert time_points and lambda schedules to numpy if needed
        if hasattr(self.time_points, 'cpu'):
            self.time_points = self.time_points.cpu().numpy()
        if hasattr(self.Λp, 'cpu'):
            self.Λp = self.Λp.cpu().numpy()
        if hasattr(self.Λm, 'cpu'):
            self.Λm = self.Λm.cpu().numpy()
        if hasattr(self.Λ_tot1, 'cpu'):
            self.Λ_tot1 = self.Λ_tot1.cpu().numpy()

        B, d   = x0.shape

        # λ–schedule
        Λ_tot1 = self.Λ_tot1

        # latent (N, M, B1)
        diff = x1 - x0
        N, M, B1 = self._sample_N_M_B1(diff, self.Λp[-1], self.Λm[-1])

        # choose time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = np.abs(self.time_points - t_target)
            k_idx_val = np.argmin(time_diffs).item()
            k_idx = np.full((B,), k_idx_val)
            t_val = self.time_points[k_idx_val]
        elif self.homogeneous_time:
            k_idx_val = np.random.randint(0, self.n_steps + 1)
            k_idx = np.full((B,), k_idx_val)
            t_val = self.time_points[k_idx_val]
        else:
            raise NotImplementedError("Non-homogeneous time is not implemented for mean-constrained bridge")

        t      = self.time_points[k_idx]                            # (B,)
        w_t    = (self.Λp[k_idx] + self.Λm[k_idx]) / Λ_tot1       # (B,)

        N_t    = np.random.binomial(N, np.expand_dims(w_t, -1))  # (B,d)
        # births up to t
        B_t = hypergeometric_numpy(N, B1, N_t)

        # ------------------------------------------------------------------
        #  Mean–constraint target  S_target
        # ------------------------------------------------------------------
        mu0  = x0.astype(float).mean(axis=0)           # (d,)
        mu1  = x1.astype(float).mean(axis=0)           # (d,)
        mu_t = (1. - t_val) * mu0 + t_val * mu1        # (d,)
        S_target = (np.round((mu_t - mu0) * B).astype(np.int64) + N_t.sum(axis=0)) // 2

        # MH projection
        B_t = mh_mean_constrained_update(
            N_s    = N_t, # .copy(),              # same time-slice
            B_s    = B_t, #.copy(),
            N_t    = N, #.copy(),
            B_t    = B1, #.copy(),
            S      = S_target,
            sweeps = self.mh_sweeps
        )
        x_t = x1 - 2 * (B1 - B_t) + (N - N_t)

        # recompute diff & slack pairs after MH (keeps N_t fixed)
        diff_t = (x_t - x0)
        M_t    = (N_t - np.abs(diff_t)) // 2

        return {
            "x0"   : x0,
            "x1"   : x1,
            "x_t"  : x_t,
            "t"    : t,          
            "M_t"    : M_t
        }

    def reverse_sampler(
        self,
        x1: np.ndarray,
        z:  dict,
        mu0: np.ndarray,
        model,
        return_trajectory: bool = False,
        return_x_hat:     bool = False,
        return_M:         bool = False,
        guidance_x0:      np.ndarray = None,
        guidance_schedule: np.ndarray = None,
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
            
        # Convert to numpy if needed
        if hasattr(x1, 'cpu'):  # if it's a torch tensor
            x1 = x1.cpu().numpy()
        if hasattr(mu0, 'cpu'):  # if it's a torch tensor
            mu0 = mu0.cpu().numpy()
            
        x_t    = x1.astype(np.int64)           # current state (starts at X₁)
        Bbatch, d = x_t.shape

        # ── forward schedule and w(t) ────────────────────────────────────
        w = (self.Λp + self.Λm) / self.Λ_tot1       # (K+1,)

        # ── initial slack-pair count at t=1 ─────────────────────────────
        λ_star = 2. * np.sqrt(self.Λp[-1] * self.Λm[-1])
        λ_star = np.broadcast_to(λ_star, x_t.shape)
        M_t    = np.random.poisson(λ_star).astype(np.int64)     # (B,d)

        # ── first model call (t=1) ──────────────────────────────────────
        t1         = np.ones_like(x_t[:, :1])
        x_t_tensor = torch.tensor(x_t).to('cuda').float()
        M_t_tensor = torch.tensor(M_t).to('cuda').float()
        t1_tensor = torch.tensor(t1).to('cuda').float()
        x0_hat_t_out   = model.sample(x_t_tensor, M_t_tensor, t1_tensor, **z).cpu().numpy()
        x0_hat_t_out   = np.round(x0_hat_t_out).astype(np.int64)

        if guidance_x0 is not None:
            guidance_mix = 1
            x0_hat_t_out =  (guidance_mix * guidance_x0 + (1 - guidance_mix) * x0_hat_t_out).round().astype(np.int64)

        weighted_dist = get_proportional_weighted_dist(x0_hat_t_out)
        x0_hat_t = sample_pert(x0_hat_t_out, weighted_dist, mu0)

        diff       = x_t - x0_hat_t
        N_t        = np.abs(diff) + 2 * M_t
        B_t        = (N_t + diff) // 2

        # target means
        mu1 = x1.astype(float).mean(axis=0) 

        # trackers
        traj, xhat_traj, M_traj = [], [], []

        # ── reverse sweep  t_k → t_{k-1} ────────────────────────────────
        for k in range(self.n_steps, 0, -1):
            ρ = (w[k-1] / w[k]).item()          # thinning prob

            # single model prediction at t_k
            t_tensor  = np.zeros_like(x_t[:, :1]) + self.time_points[k-1] #.cpu().numpy()
            x_t_tensor = torch.tensor(x_t).to('cuda').float()
            M_t_tensor = torch.tensor(M_t).to('cuda').float()
            t_tensor = torch.tensor(t_tensor).to('cuda').float()
            x0_hat_t_out  = model.sample(x_t_tensor, M_t_tensor, t_tensor, **z).cpu().numpy()
            x0_hat_t_out  = np.round(x0_hat_t_out).astype(np.int64)

            if guidance_x0 is not None:
                guidance_mix = guidance_schedule[k-1]
                x0_hat_t_out = (guidance_mix * guidance_x0 + (1 - guidance_mix) * x0_hat_t_out).round().astype(np.int64)

            weighted_dist = get_proportional_weighted_dist(x0_hat_t_out)
            x0_hat_t = sample_pert(x0_hat_t_out.astype(np.int64), weighted_dist, mu0 - x0_hat_t_out.mean(axis=0))

            print(np.max(np.abs(x0_hat_t_out - x0_hat_t) / (x0_hat_t_out + 1)), (mu0 - x0_hat_t_out.mean(axis=0)).max(), (mu0-x0_hat_t.mean(axis=0)).max())

            # print("max proj diff",np.max(x0_hat_t.mean(axis=0) - mu0))
            # print(M_t.min())
            diff      = x_t - x0_hat_t
            N_t       = (np.abs(diff) + 2 * M_t).astype(np.int64)
            B_t       = (N_t + diff) // 2

            non_zero = N_t > 0
            N_s = np.zeros_like(N_t)
            N_s[non_zero] = np.random.binomial(N_t[non_zero], ρ)

            # births that survive the thinning  ~ Hypergeom
            B_s = np.zeros_like(B_t)
            B_s[non_zero] = hypergeometric(
                total_count   = N_t[non_zero],
                success_count = B_t[non_zero],
                num_draws     = N_s[non_zero]
            )

            # Metropolis–Hastings mean-projection
            t_s   = self.time_points[k-1].item()
            mu_s  = (1. - t_s) * mu0 + t_s * mu1          # (d,)
            mu_curr = x_t.mean(axis=0)
            S_target = (np.round((mu_s - mu_curr) * Bbatch).astype(np.int64) + N_s.sum(axis=0)) // 2
            # print("target diff", np.max(x_t.mean(axis=0) + 2 * S_target / Bbatch - N_s.mean(axis=0) - mu_s), t_s, k)
            # print("x0", x0_hat_t.mean(axis=0),mu0, (x0_hat_t == 0).mean())

            B_s = mh_mean_constrained_update( 
                N_s = N_s.copy(),
                B_s = B_s.copy(),
                N_t = N_t.copy(),
                B_t = B_t.copy(),
                S = S_target,
                sweeps = self.mh_sweeps,
                override_support=(k == 1)
            )

            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s) # x0_hat_t + 2 * B_s - N_s
            # print("post", np.max(x_s.mean(axis=0) - mu_s), x_s.mean(axis=0), mu_s, t_s, k)

            # derive new slack pairs AFTER MH projection
            diff_s = np.abs(x_s - x0_hat_t)
            M_s    = ((N_s - diff_s) // 2)                # = min{B_s, D_s}

            # roll forward
            x_t, M_t = x_s, M_s
            N_t, B_t = N_s, ((N_s + (x_s - x0_hat_t)) // 2)

            if return_trajectory: traj.append(x_s.copy())
            if return_x_hat:      xhat_traj.append(x0_hat_t.copy())
            if return_M:          M_traj.append(M_s.copy())

        # ── output ──────────────────────────────────────────────────────
        outs = [x_t]
        if return_trajectory: outs.append(traj)
        if return_x_hat:      outs.append(xhat_traj)
        if return_M:          outs.append(M_traj)
        return tuple(outs) if len(outs) > 1 else x_t 