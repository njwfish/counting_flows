import numpy as np
import torch
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from ...sampling.hypergeom import hypergeometric


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
        
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

        self.lam_p, self.lam_m, self.Λp, self.Λm = make_lambda_schedule(
            self.time_points, self.lam0, self.lam1, schedule_type
        )
        self.Λ_tot1 = self.Λp[-1] + self.Λm[-1]


    def _sample_N_B1(self, diff: np.ndarray, Lambda_p: float, Lambda_m: float):
        """
        diff = x1 - x0   (shape [B, d])
        Lambda_p, Lambda_m: Final cumulative integrals from lambda schedule (scalars)
        Returns N (total jumps) and B1 (birth count at t=1).
        Uses Poisson for total jumps with lambda_star from schedule.
        """
        B, d = diff.shape
        lambda_star = 2.0 * np.sqrt(Lambda_p * Lambda_m)
        lambda_star = np.broadcast_to(lambda_star, diff.shape)
        M = np.random.poisson(lambda_star).astype(np.int64)
        N = np.abs(diff) + 2 * M
        B1 = (N + diff) // 2
        return N, M, B1

    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        valid_idx = np.random.randint(1, len(self.time_points) - 1)
        t = self.time_points[valid_idx]
        t_idx = valid_idx
        return t, t_idx

    def __call__(self, batch, t_target=None):
        if isinstance(batch, list):
            x0 = np.stack([b["x0"] for b in batch])
            x1 = np.stack([b["x1"] for b in batch])
            z = np.stack([b["z"]  for b in batch])
        else:
            x0 = batch["x0"]
            x1 = batch["x1"]
            z = batch["z"]
        
        B, d = x0.shape

        diff = (x1 - x0)
        N, M, B1 = self._sample_N_B1(diff, self.Λp[-1], self.Λm[-1])

        if t_target is not None:
            time_diffs = np.abs(self.time_points - t_target)
            k_idx_scalar = np.argmin(time_diffs).item()
            k_idx = np.full((B,), k_idx_scalar)
        elif self.homogeneous_time:
            k_idx_scalar = np.random.randint(0, self.n_steps + 1)
            k_idx = np.full((B,), k_idx_scalar)
        else:
            k_idx = np.random.randint(1, self.n_steps + 1, (B,))
            
        t     = self.time_points[k_idx]
        w_t   = (self.Λp[k_idx] + self.Λm[k_idx]) / self.Λ_tot1
        
        N_t = np.random.binomial(N, np.expand_dims(w_t, -1))

        B_t = hypergeometric(
            total_count=N,
            success_count=B1,
            num_draws=N_t
        )

        x_t = x1 - 2 * (B1 - B_t) + (N - N_t)

        diff_t = np.abs(x_t - x0)
        M_t    = ((N_t - diff_t) >> 1)

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "M" : M_t, 
        }
    
    def reverse_sampler(
        self,
        x1:  np.ndarray,
        z:   np.ndarray,
        model,
        return_trajectory: bool = False,
        return_x_hat:     bool = False,
        return_M:         bool = False,
        x0 = None,
    ):
        x_t     = x1.astype(np.int64)
        B, d    = x_t.shape

        w  = (self.Λp + self.Λm) / self.Λ_tot1

        lambda_star = 2. * np.sqrt(self.Λp[-1] * self.Λm[-1])
        lambda_star = np.broadcast_to(lambda_star, x_t.shape)
        M_t    = np.random.poisson(lambda_star).astype(np.int64)

        t1          = np.ones_like(x_t[:, :1])
        x_t_tensor = torch.from_numpy(x_t).float()
        M_t_tensor = torch.from_numpy(M_t).float()
        z_tensor = torch.from_numpy(z).float()
        t1_tensor = torch.from_numpy(t1).float()
        
        x0_hat_t    = model.sample(x_t_tensor, M_t_tensor, z_tensor, t1_tensor).cpu().numpy().round().astype(np.int64)
        diff        = x_t - x0_hat_t
        N_t         = np.abs(diff) + 2 * M_t
        B_t         = (N_t + diff) >> 1

        traj, xhat_traj, M_traj = [], [], []

        for k in range(self.n_steps, 0, -1):
            ρ = (w[k-1] / w[k]).item()

            t_tensor  = np.zeros_like(x_t[:, :1]) + self.time_points[k-1]
            x_t_tensor = torch.from_numpy(x_t).float()
            M_t_tensor = torch.from_numpy(M_t).float()
            z_tensor = torch.from_numpy(z).float()
            t_tensor = torch.from_numpy(t_tensor).float()

            x0_hat_t = model.sample(x_t_tensor, M_t_tensor, z_tensor, t_tensor).cpu().numpy().round().astype(np.int64)

            diff  = x_t - x0_hat_t
            N_t   = np.abs(diff) + 2 * M_t
            B_t   = (N_t + diff) >> 1

            non_zero = N_t > 0
            N_s = np.zeros_like(N_t)
            N_s[non_zero] = np.random.binomial(N_t[non_zero], ρ)

            B_s = np.zeros_like(B_t)
            B_s[non_zero] = hypergeometric(
                total_count   = N_t[non_zero],
                success_count = B_t[non_zero],
                num_draws     = N_s[non_zero]
            )

            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)
            diff_s = np.abs(x_s - x0_hat_t)
            M_s = (N_s - diff_s) // 2  

            x_t, M_t = x_s, M_s

            if return_trajectory: traj.append(x_t)
            if return_x_hat:     xhat_traj.append(x0_hat_t)
            if return_M:         M_traj.append(M_t)

        outs = [x_t]
        if return_trajectory: outs.append(traj)
        if return_x_hat:     outs.append(xhat_traj)
        if return_M:         outs.append(M_traj)
        return tuple(outs) if len(outs) > 1 else x_t 