import torch
from torch.distributions import Poisson, Binomial, Beta, NegativeBinomial
import numpy as np
from .scheduling import make_time_spacing_schedule, make_phi_schedule, make_lambda_schedule


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
        lambda_star = 2.0 * torch.sqrt(Lambda_p * Lambda_m)        # scalar
        lambda_star = lambda_star.unsqueeze(-1).expand_as(diff)    # shape (B, d)
        M = torch.poisson(lambda_star).long().to(device)
        N = diff.abs() + 2 * M
        B1 = (N + diff) // 2   # integer division
        return N, B1

    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx
        return t, t_idx

    def __call__(self, batch):
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
        N, B1 = self._sample_N_B1(diff, Λp[-1], Λm[-1], device)       # shape (B, d)

        # pick random interior k (avoid k=0,K)
        if self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx_scalar = torch.randint(1, self.n_steps + 1, (1,), device=device).item()
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

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "N"   : N,
            "B1"  : B1,
            "grid": grid,
            "Λp"  : Λp,
            "Λm"  : Λm,
        }


class PolyaBDBridgeCollate:
    """
    Collate for Polya birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    Uses NegativeBinomial for total jumps instead of Poisson.
    """

    def __init__(
        self,
        n_steps: int,
        r: float = 1.0,
        beta: float = 1.0,
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
        self.r = r
        self.beta = beta
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.sched_kwargs = schedule_kwargs
        
        # Create time spacing
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

    def _sample_N_B1(self, diff: torch.Tensor, device):
        """
        diff = x1 - x0   (shape [B, d])
        Returns N (total jumps) and B1 (birth count at t=1).
        Uses NegativeBinomial for total jumps.
        """
        B, d = diff.shape
        #  M ~ NB(r, beta/(beta+1))
        nb = NegativeBinomial(total_count=self.r, probs=self.beta / (self.beta + 1))
        M = nb.sample((B, d)).long().to(device)
        N = diff.abs() + 2 * M
        B1 = (N + diff) // 2   # integer division
        return N, B1

    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx
        return t, t_idx

    def __call__(self, batch):
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
        N, B1 = self._sample_N_B1(diff, device)       # shape (B, d)

        # pick random interior k (avoid k=0,K)
        if self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx_scalar = torch.randint(1, self.n_steps, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        else:
            # Sample different time indices for each sample (original behavior)
            k_idx = torch.randint(1, self.n_steps, (B,), device=device)
            
        t     = grid[k_idx]                               # (B,)
        w_t   = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1          # (B,)
        # N_t   = torch.round(N.float() * w_t.unsqueeze(-1)).long()       # (B, d)
        N_t = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()

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

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "N"   : N,
            "B1"  : B1,
            "grid": grid,
            "Λp"  : Λp,
            "Λm"  : Λm,
        }
