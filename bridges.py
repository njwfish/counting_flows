import torch

import numpy as np
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from .sampling.hypergeom import manual_hypergeometric
from .sampling.mean_constrained import mh_mean_constrained_update


class PoissonBDBridgeCollate:
    """
    Collate for Poisson birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,
        lam_p1: float = 8.0,
        lam_m0: float = 8.0,
        lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.sched_kwargs = schedule_kwargs
        
        # Create time spacing
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

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

        # schedules
        grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
            self.n_steps, self.lam_p0, self.lam_p1, self.lam_m0, self.lam_m1,
            self.schedule_type, device=device
        )
        Λ_tot1 = Λp[-1] + Λm[-1]

        # latent (N,B1)
        diff = (x1 - x0)          # (B, d) - don't squeeze
        N, M, B1 = self._sample_N_B1(diff, Λp[-1], Λm[-1], device)       # shape (B, d)

        # pick time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = torch.abs(grid - t_target)
            k_idx_scalar = torch.argmin(time_diffs).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        elif self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx_scalar = torch.randint(0, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        else:
            # Sample different time indices for each sample (original behavior)
            k_idx = torch.randint(1, self.n_steps + 1, (B,), device=device)
            
            
        t     = grid[k_idx]                               # (B,)
        w_t   = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1          # (B,)
        N_t   = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()       # (B, d)

        # draw B_t ~ Hypergeom(N, B1, N_t)
        B_t = torch.empty_like(N)
        
        # Vectorized hypergeometric sampling
        B_t = manual_hypergeometric(
            total_count=N, # .cpu().numpy(),
            success_count=B1, # .cpu().numpy(),
            num_draws=N_t # .cpu().numpy()
        )
        # B_t = torch.from_numpy(B_t_np).to(device)

        x_t = x0 + (2 * B_t - N_t)          # (B, d)

        diff_t = (x_t - x0).abs()          # (B,d)
        M_t    = ((N_t - diff_t) >> 1)     # integer tensor (B,d)

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "N"   : N,
            "M" : M_t, 
            "B1"  : B1,
            "grid": grid,
            "Λp"  : Λp,
            "Λm"  : Λm,
        }

class PoissonMeanConstrainedBDBridgeCollate:
    """
    Same latent construction as PoissonBDBridgeCollate but the intermediate
    state x_t is *resampled* (via a fast MH swap) so that
            sum_i x_t^{(i)}  =  d * [ (1–t)·mean(x0) + t·mean(x1) ] .
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,  lam_p1: float = 8.0,
        lam_m0: float = 8.0,  lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        mh_sweeps: int = 5,
        **schedule_kwargs,
    ):
        self.n_steps        = n_steps
        self.lam_p0, self.lam_p1 = lam_p0, lam_p1
        self.lam_m0, self.lam_m1 = lam_m0, lam_m1
        self.schedule_type  = schedule_type
        self.homogeneous_time = homogeneous_time
        self.time_points    = make_time_spacing_schedule(
                                  n_steps, time_schedule, **schedule_kwargs)
        self.mh_sweeps      = mh_sweeps
        self.sched_kwargs   = schedule_kwargs

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

        # λ–schedule
        grid, _, _, Λp, Λm = make_lambda_schedule(
            self.n_steps,
            self.lam_p0, self.lam_p1,
            self.lam_m0, self.lam_m1,
            self.schedule_type,
            device=device,
        )
        Λ_tot1 = Λp[-1] + Λm[-1]

        # latent (N, M, B1)
        diff = x1 - x0
        N, M, B1 = self._sample_N_M_B1(diff, Λp[-1], Λm[-1], device)

        # choose time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = torch.abs(grid - t_target)
            k_idx_val = torch.argmin(time_diffs).item()
            k_idx = torch.full((B,), k_idx_val, device=device)
            t_val = grid[k_idx_val]
        elif self.homogeneous_time:
            k_idx_val = torch.randint(0, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_val, device=device)
            t_val = grid[k_idx_val]
        else:
            raise NotImplementedError("Non-homogeneous time is not implemented for mean-constrained bridge")

        t      = grid[k_idx]                            # (B,)
        w_t    = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1       # (B,)
        N_t    = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()  # (B,d)

        # births up to t
        B_t = manual_hypergeometric(N, B1, N_t)

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
            "t"    : t,          # (B,)
            "z"    : z,
            "N"    : N,
            "M"    : M_t,
            "B1"   : B1,
            "grid" : grid,
            "Λp"   : Λp,
            "Λm"   : Λm,
        }



class ReflectedPoissonBDBridgeCollate:
    """
    Collate for *reflected* Poisson birth–death bridge (equal λ₊,λ₋ at each time).
    Samples x_t = |X_t| by drawing X_t via Hypergeometric(N, B1, N_t) then taking abs.
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,
        lam_p1: float = 8.0,
        lam_m0: float = 8.0,
        lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        # precompute time grid for optional use
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)
