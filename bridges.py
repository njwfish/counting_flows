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

class PoissonBridgeCollate:
    """Collate function for Poisson bridge sampling with time scheduling"""
    
    def __init__(self, n_steps, time_schedule="uniform", homogeneous_time=True, **schedule_kwargs):
        """
        Args:
            n_steps: Number of diffusion steps
            time_schedule: Time point distribution
            homogeneous_time: If True, all samples in batch get same time
            **schedule_kwargs: Additional schedule parameters
        """
        self.n_steps = n_steps
        self.homogeneous_time = homogeneous_time
        
        # Always use make_time_spacing_schedule (handles uniform too)
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)
    
    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx  # Use the actual index in the schedule
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

        # sample time
        t, t_idx = self._sample_time()
        t = torch.full((B,), t, device=device)

        # for poisson bridge we sample x_t|x_0,x_1 from a hypergeometric
        # which requires integers
        x0_int = x0.long()
        x1_int = x1.long()
        x_min = torch.minimum(x0_int, x1_int)
        x_max = torch.maximum(x0_int, x1_int)

        # sample x_t ~ Hypergeometric(x_max, x_min, round(x_max * t))
        n_draws = torch.round(x_max.float() * t.unsqueeze(1)).long()
        n_draws = torch.clamp(n_draws, 0, x_max)  # ensure valid range

        # use manual hypergeometric sampling
        x_t_np = manual_hypergeometric(
            total_count=x_max.cpu().numpy(),
            num_successes=x_min.cpu().numpy(),
            num_draws=n_draws.cpu().numpy()
        )
        x_t = torch.from_numpy(x_t_np).long().to(device)

        # for decreasing case, we need to adjust
        decreasing_mask = x0_int > x1_int
        x_t_adj = x_t.clone()
        x_t_adj[decreasing_mask] = x0_int[decreasing_mask] - (x_t[decreasing_mask] - x1_int[decreasing_mask])
        x_t = x_t_adj

        return {
            "x0": x0_int.float(),
            "x1": x1_int.float(),
            "x_t": x_t.float(),
            "t": t,
            "z": z,
        }


class NBBridgeCollate:
    """Collate function for Negative Binomial (Polya) bridge sampling with time scheduling"""
    
    def __init__(self, n_steps, r_min=1.0, r_max=20.0, r_schedule="linear", 
                 time_schedule="uniform", homogeneous_time=True, **schedule_kwargs):
        """
        Args:
            n_steps: Number of diffusion steps
            r_min, r_max: Range for r(t) schedule
            r_schedule: Schedule type for r(t)
            time_schedule: Time point distribution
            homogeneous_time: If True, all samples in batch get same time
            **schedule_kwargs: Additional schedule parameters
        """
        self.n_steps = n_steps
        self.homogeneous_time = homogeneous_time
        
        # Create Φ(t) schedule using proper cumulative integration
        self.phi_sched, self.R = make_phi_schedule(n_steps, r_min, r_max, r_schedule, **schedule_kwargs)
        
        # Always use make_time_spacing_schedule (handles uniform too)
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)
    
    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx  # Use the actual index in the schedule
        return t, t_idx
    
    def __call__(self, batch):
        # batch = list of dicts with keys 'x0','x1','z'
        x0 = torch.stack([b["x0"] for b in batch])
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
        
        # Handle case where dataset already produces batches (extra dimension from DataLoader)
        if len(x0.shape) == 3:  # (1, B, d)
            x0 = x0.squeeze(0)  # (B, d)
            x1 = x1.squeeze(0)  # (B, d)
            z = z.squeeze(0)    # (B, context_dim)
        
        device = x0.device
        batch_size = x0.shape[0]
        
        if self.homogeneous_time:
            # Sample one time for the entire batch
            t, t_idx = self._sample_time()
            t_batch = torch.full((batch_size,), t, dtype=torch.float32, device=device)
            t_idx_list = [t_idx] * batch_size  # Same t_idx for all samples
        else:
            # Sample time for each item in batch (original behavior)
            t_list = []
            t_idx_list = []
            for _ in range(batch_size):
                t, t_idx = self._sample_time()
                t_list.append(t)
                t_idx_list.append(t_idx)
            
            t_batch = torch.tensor(t_list, dtype=torch.float32, device=device)
        
        # Get Φ(t) values for each sample in batch (proper bridge parameters)
        phi_t_batch = torch.tensor([self.phi_sched[t_idx].item() for t_idx in t_idx_list], 
                                 dtype=torch.float32, device=device)
        
        # Batch Beta-Binomial bridge sampling using proper formulation
        n = (x1 - x0).abs().long()
        
        # Proper alpha/beta computation: α_t = Φ(t), β_t = R - Φ(t)
        alpha = torch.clamp(phi_t_batch.unsqueeze(-1), min=1e-3)
        beta = torch.clamp((self.R - phi_t_batch).unsqueeze(-1), min=1e-3)
        
        # Sample p from Beta distribution
        p = Beta(alpha, beta).sample().clamp(1e-6, 1-1e-6).to(device)
        
        # Sample k from Binomial
        k = Binomial(total_count=n, probs=p).sample().to(device)
        x_t = x0 + torch.sign(x1 - x0) * k
        
        return {
            'x0': x0,
            'x1': x1,
            'x_t': x_t, 
            't': t_batch,
            'z': z,
            'r': phi_t_batch  # Now stores Φ(t) values for consistency
        }


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
            k_idx_scalar = torch.randint(1, self.n_steps, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        else:
            # Sample different time indices for each sample (original behavior)
            k_idx = torch.randint(1, self.n_steps, (B,), device=device)
            
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


class ReflectedBDBridgeCollate:
    """
    Collate for training with a reflected (non-negative) birth–death bridge
    under the equal‐rate assumption λ₊(t)=λ₋(t).
    """

    def __init__(
        self,
        n_steps: int,
        lam0: float = 8.0,
        lam1: float = 8.0,
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        """
        Args:
          n_steps: number of grid points in [0,1]
          lam0, lam1: birth=death rates at t=0 and t=1
          time_schedule: "uniform" | "linear" | …
          homogeneous_time: If True, all samples in batch get same time
        """
        self.n_steps = n_steps
        self.homogeneous_time = homogeneous_time

        # 1) time grid t_k ∈ [0,1]
        self.time_points = make_time_spacing_schedule(
            n_steps, time_schedule, **schedule_kwargs
        )  # (K,)

        # 2) λ(t_k) = lam0 + (lam1−lam0)*t_k
        self.lam_vals = lam0 + (lam1 - lam0) * self.time_points  # (K,)

        # 3) cumulative integral Λ(t_k) via trapezoid rule
        dt = 1.0 / (n_steps - 1)
        lam_mid = 0.5 * (self.lam_vals[:-1] + self.lam_vals[1:])
        self.Lam = torch.zeros_like(self.lam_vals)
        self.Lam[1:] = torch.cumsum(lam_mid * dt, dim=0)  # (K,)

    def _sample_time_idx(self, batch_size: int):
        """
        Randomly choose an interior index k ∈ {1,...,K-1} for each of B samples.
        """
        if self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx = torch.randint(1, self.n_steps - 1, size=(1,)).item()
            return torch.full((batch_size,), k_idx)
        else:
            # Sample different time indices for each sample (original behavior)
            return torch.randint(1, self.n_steps - 1, size=(batch_size,))

    def __call__(self, batch):
        # batch = list of dicts with keys 'x0','x1','z'
        x0 = torch.stack([b["x0"] for b in batch])
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
        
        # Handle case where dataset already produces batches (extra dimension from DataLoader)
        if len(x0.shape) == 3:  # (1, B, d)
            x0 = x0.squeeze(0)  # (B, d)
            x1 = x1.squeeze(0)  # (B, d)
            z = z.squeeze(0)    # (B, context_dim)
        
        device = x0.device
        B, d = x0.shape

        # 1) pick random interior time‐index k
        k_idxs = self._sample_time_idx(B)  # (B,) - keep on CPU for indexing
        t_vals = self.time_points[k_idxs].to(device)  # (B,)

        # 2) final integral Λ(1)
        Lambda1 = self.Lam[-1].item()  # scalar

        # 3) latent total jumps N at t=1:  N = |x1-x0| + 2*M,  M ~ Pois(2*Λ(1))
        diff = (x1 - x0).abs()  # (B,d)
        lam_star = 2.0 * Lambda1  # scalar
        M = Poisson(lam_star * torch.ones_like(diff, dtype=torch.float32)).sample().long().to(device)  # (B,d)
        N = diff + 2 * M  # (B,d)

        # 4) signed births at t=1: B1 = (N + (x1-x0)) // 2
        B1 = (N + (x1 - x0)) // 2  # (B,d)

        # 5) interior total N_t = floor(N * w(t)),  w(t)=Λ(t)/Λ(1)
        Lam_t = self.Lam[k_idxs].unsqueeze(-1).to(device)  # (B,1)
        w_t   = (Lam_t / Lambda1).clamp(0.0, 1.0)          # (B,1)
        N_t   = torch.floor(N.float() * w_t).long()        # (B,d)

        # 6) draw B_t ~ Hypergeom(N, B1, N_t)
        #    i.e. from the unreflected scheme.  We'll reflect later.
        B_t = torch.zeros_like(N_t)  # (B,d)
        N_np   = N.cpu().numpy()
        B1_np  = B1.cpu().numpy()
        Nt_np  = N_t.cpu().numpy()
        out    = np.zeros_like(Nt_np)

        success_clipped = np.minimum(B1_np, N_np)
        draws_clipped   = np.minimum(Nt_np, N_np)

        mask_valid = (draws_clipped > 0) & (success_clipped > 0)
        if mask_valid.any():
            out[mask_valid] = np.random.hypergeometric(
                success_clipped[mask_valid],
                (N_np[mask_valid] - success_clipped[mask_valid]),
                draws_clipped[mask_valid]
            )
        B_t = torch.from_numpy(out).to(device)

        # 7) compute unreflected X_t^raw = x0 + (2*B_t - N_t), then reflect
        x_t = x0 + (2 * B_t - N_t)
        x_t = x_t.abs().long()  # reflected at zero

        return {
            'x0' : x0,     # (B,d)
            'x1' : x1,     # (B,d)
            'x_t': x_t,    # (B,d)
            't'  : t_vals, # (B,)
            'N'  : N,      # (B,d)
            'B1' : B1,     # (B,d)
            'N_t': N_t,    # (B,d)
            'B_t': B_t,    # (B,d)
            'z'  : z       # (B,d)
        }