import torch
from torch.distributions import NegativeBinomial
import numpy as np
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from torch.distributions import Multinomial

def manual_hypergeometric(total_count, num_successes, num_draws):
    """
    Vectorized implementation of hypergeometric sampling using numpy broadcasting.
    
    Args:
        total_count: Total population size (array-like)
        num_successes: Number of success items in population (array-like)
        num_draws: Number of items drawn (array-like)
    
    Returns:
        Number of successes in the drawn sample (same shape as inputs)
    """
    # Convert to numpy arrays for vectorized operations
    total_count = np.asarray(total_count)
    num_successes = np.asarray(num_successes)
    num_draws = np.asarray(num_draws)
    
    # Bounds checking to prevent invalid hypergeometric parameters
    num_successes = np.clip(num_successes, 0, total_count)
    num_draws = np.clip(num_draws, 0, total_count)
    
    # Handle edge cases
    mask_zero_draws = (num_draws == 0)
    mask_zero_successes = (num_successes == 0)
    mask_draws_ge_total = (num_draws >= total_count)
    mask_successes_ge_total = (num_successes >= total_count)
    
    # Use numpy's vectorized hypergeometric
    result = np.zeros_like(num_draws)
    
    # Only call hypergeometric for valid cases
    valid_mask = ~(mask_zero_draws | mask_zero_successes | mask_draws_ge_total | mask_successes_ge_total)
    
    if np.any(valid_mask):
        result[valid_mask] = np.random.hypergeometric(
            num_successes[valid_mask], 
            total_count[valid_mask] - num_successes[valid_mask], 
            num_draws[valid_mask]
        )
    
    # Handle edge cases
    result[mask_zero_draws] = 0
    result[mask_zero_successes] = 0
    result[mask_draws_ge_total] = num_successes[mask_draws_ge_total]
    result[mask_successes_ge_total] = num_draws[mask_successes_ge_total]
    
    return result

# -----------------------------------------------------------------------------
# 1) Sum‐constrained Multinomial proposal
# -----------------------------------------------------------------------------

import torch
from torch.distributions import Multinomial
from typing import Tuple

def hypergeom_logpmf(k,N,K,n):
    return (
        torch.lgamma(K+1) - torch.lgamma(k+1) - torch.lgamma(K-k+1)
      + torch.lgamma(N-K+1) - torch.lgamma(n-k+1)
      - torch.lgamma(N-K-n+k+1)
      - (torch.lgamma(N+1) - torch.lgamma(n+1) - torch.lgamma(N-n+1))
    )

from torch.distributions import Multinomial, Binomial

def constrained_multinomial_proposal(
    x:     torch.LongTensor,    # (B,d)
    x_exp: torch.Tensor,        # (B,d)
    S:     torch.LongTensor,    # (d,) target *column*-sums
    a:     torch.LongTensor,    # (B,d) current a = x0_hat_t
    N_s:   torch.LongTensor,    # (B,d) latent total jumps at s
):
    """
    Propose x' with column-sums S by distributing
    R_j = S_j - sum_b x_{b,j} via a signed multinomial
    across the B rows of each column j.
    Then clip so we never leave the bridge support:
      |x'_{b,j} - a_{b,j}| <= N_s_{b,j}.
    """
    B, d = x.shape
    # 1) compute column residuals
    col_sum = x.sum(dim=0)            # (d,)
    R       = S - col_sum            # (d,)
    sgn     = R.sign().long()         # (d,)
    Rabs    = R.abs().long()          # (d,)

    # 2) directional weights
    diff    = (x_exp - x).float() * sgn.unsqueeze(0)   # (B,d)
    weights = diff.clamp(min=0.0)                     # zero out opp. dir
    wsum    = weights.sum(dim=0, keepdim=True)        # (1,d)
    # avoid all-zero
    weights[:, wsum[0]==0] += 1.0
    wsum    = weights.sum(dim=0, keepdim=True)        # updated
    probs   = weights / wsum                          # (B,d)

    # 3) per-column Multinomial into B rows
    delta = torch.zeros_like(x)
    for j in range(d):
        if Rabs[j] > 0:
            m = Multinomial(total_count=Rabs[j].item(),
                             probs=probs[:,j]).sample()  # (B,)
            delta[:,j] = m.round().long()

    # 4) raw proposal
    x_prop = x + sgn.unsqueeze(0) * delta             # (B,d)

    # 5) clip each entry to remain in support:
    #    a_{b,j} - N_s_{b,j} <= x_prop_{b,j} <= a_{b,j} + N_s_{b,j}
    lower = a - N_s
    upper = a + N_s
    x_prop = torch.max(torch.min(x_prop, upper), lower)

    return x_prop, sgn

@torch.no_grad()
def mh_mean_constrained_update(
    x:       torch.LongTensor,    # (B,d) must sum columns to S
    x_exp:   torch.Tensor,        # (B,d) expected x
    S:       torch.LongTensor,    # (d,) column-sum target
    a:       torch.LongTensor,    # (B,d) x0_hat_t
    N_t:     torch.LongTensor,    # (B,d) latents at t
    B_t:     torch.LongTensor,    # (B,d) latents at t
    N_s:     torch.LongTensor,    # (B,d) latents at s
    sweeps:  int = 10
) -> torch.LongTensor:
    """
    MH on the *column*-sum constrained reverse kernel.
    """
    B, d = x.shape
    device = x.device

    # 0) project once onto correct column-sum manifold
    x, _ = constrained_multinomial_proposal(x, x_exp, S, a, N_s)

    for _ in range(sweeps):
        # 1) pick a column j and two distinct rows p,q
        j = torch.randint(d, (1,), device=device).item()
        p = torch.randint(B, (1,), device=device).item()
        q = torch.randint(B, (1,), device=device).item()
        if p == q:
            continue

        # 2) propose to move +1 at (p,j), -1 at (q,j)
        x_prop = x.clone()
        x_prop[p,j] += 1
        x_prop[q,j] -= 1

        # 3) quick support check
        if not (0 <= x_prop[q,j]):
            continue
        for idx in (p,q):
            if abs(x_prop[idx,j] - a[idx,j]) > N_s[idx,j]:
                break
        else:
            # 4) compute per-coordinate hypergeom log-pmf ratios
            #    only rows p and q in column j change
            def HG_log(b, Nt, Bt, Ns):
                return hypergeom_logpmf(
                    k=b,
                    N=Nt, K=Bt, n=Ns
                )
            # current births:
            Bsc_p = (N_s[p,j] + (x[p,j]-a[p,j]))//2
            Bsc_q = (N_s[q,j] + (x[q,j]-a[q,j]))//2
            # proposed births:
            Bsp_p = (N_s[p,j] + (x_prop[p,j]-a[p,j]))//2
            Bsp_q = (N_s[q,j] + (x_prop[q,j]-a[q,j]))//2

            logp_curr = ( 
                HG_log(Bsc_p, N_t[p,j], B_t[p,j], N_s[p,j])
              + HG_log(Bsc_q, N_t[q,j], B_t[q,j], N_s[q,j])
            )
            logp_prop = (
                HG_log(Bsp_p, N_t[p,j], B_t[p,j], N_s[p,j])
              + HG_log(Bsp_q, N_t[q,j], B_t[q,j], N_s[q,j])
            )

            alpha = (logp_prop - logp_curr).exp().clamp(max=1.0)
            if torch.rand(1, device=device) < alpha:
                x = x_prop

    return x



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
        B_t_np = manual_hypergeometric(
            total_count=N.cpu().numpy(),
            num_successes=B1.cpu().numpy(),
            num_draws=N_t.cpu().numpy()
        )
        B_t = torch.from_numpy(B_t_np).to(device)

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
        B_t = torch.from_numpy(
                  manual_hypergeometric(N.cpu().numpy(),
                                        B1.cpu().numpy(),
                                        N_t.cpu().numpy())
              ).to(device)

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
