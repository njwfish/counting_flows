"""
PyTorch Datasets for Count-based Flow Matching

Provides two core dataset classes for count flow scenarios:
1. PoissonDataset: Poisson endpoints with configurable base measure
2. BetaBinomialDataset: BetaBinomial endpoints with small alpha/beta (tricky distribution)

Each dataset can use either fixed or random base measures.
Time scheduling is handled by collate functions, not datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Poisson, Binomial, Beta, Gamma, NegativeBinomial, Categorical
import numpy as np
from .scheduling import make_time_spacing_schedule, make_phi_schedule, make_lambda_schedule
from abc import ABC, abstractmethod


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


class BaseCountDataset(Dataset, ABC):
    """Base class for count flow datasets"""
    
    def __init__(self, size, d, fixed_base=False, batch_size=None, homogeneous=False):
        """
        Args:
            size: Dataset size (number of samples per epoch)
            d: Dimensionality of count vectors
            fixed_base: If True, use fixed base measure; if False, sample randomly
            batch_size: If provided, __getitem__ returns batches of this size
            homogeneous: If True, all samples in batch share same parameters
        """
        self.size = size
        self.d = d
        self.fixed_base = fixed_base
        self.batch_size = batch_size
        self.homogeneous = homogeneous
        
        # Generate fixed base measure if needed
        if self.fixed_base:
            self.base_params = self._generate_fixed_base()
    
    def __len__(self):
        if self.batch_size is not None:
            return self.size // self.batch_size
        return self.size
    
    def __getitem__(self, idx):
        """Generate endpoint pair or batch"""
        if self.batch_size is not None:
            return self._generate_batch(self.batch_size, self.homogeneous)
        else:
            x0, x1, z = self._generate_endpoints()
            return {
                'x0': x0.long(),
                'x1': x1.long(),
                'z': z.float()
            }
    
    def _generate_batch(self, batch_size, homogeneous=False):
        """Generate a batch of endpoint pairs"""
        if homogeneous:
            # Generate one set of parameters and repeat for the batch
            x0_single, x1_single, z_single = self._generate_endpoints()
            
            x0_batch = x0_single.unsqueeze(0).repeat(batch_size, 1)
            x1_batch = x1_single.unsqueeze(0).repeat(batch_size, 1)
            z_batch = z_single.unsqueeze(0).repeat(batch_size, 1)
        else:
            # Generate different parameters for each sample
            x0_list, x1_list, z_list = [], [], []
            for _ in range(batch_size):
                x0, x1, z = self._generate_endpoints()
                x0_list.append(x0)
                x1_list.append(x1)
                z_list.append(z)
            
            x0_batch = torch.stack(x0_list)
            x1_batch = torch.stack(x1_list)
            z_batch = torch.stack(z_list)
        
        return {
            'x0': x0_batch.long(),
            'x1': x1_batch.long(),
            'z': z_batch.float()
        }
    
    @abstractmethod
    def _generate_endpoints(self):
        """Generate x0, x1, z for this dataset type"""
        pass
    
    @abstractmethod
    def _generate_fixed_base(self):
        """Generate fixed base measure parameters"""
        pass
    
    @abstractmethod
    def get_context_dim(self):
        """Return context dimension for this dataset"""
        pass


class PoissonDataset(BaseCountDataset):
    """
    Poisson Dataset: x0 ~ Poisson(λ₀), x1 ~ Poisson(λ₁)
    
    - If fixed_base=True: λ₀ is fixed, λ₁ is random
    - If fixed_base=False: Both λ₀ and λ₁ are random
    """
    
    def __init__(self, size, d, lam_scale=50.0, fixed_base=False, fixed_lam=10.0, 
                 batch_size=None, homogeneous=False):
        self.lam_scale = lam_scale
        self.fixed_lam = fixed_lam
        super().__init__(size, d, fixed_base, batch_size, homogeneous)
        
    def _generate_fixed_base(self):
        """Generate fixed λ₀ for all samples"""
        return torch.full((self.d,), self.fixed_lam, dtype=torch.float32)
    
    def _generate_endpoints(self):
        if self.fixed_base:
            # Fixed λ₀, random λ₁
            lam0 = self.base_params
            lam1 = self.lam_scale * torch.rand(self.d)
        else:
            # Both random
            lam0 = self.lam_scale * torch.rand(self.d)
            lam1 = self.lam_scale * torch.rand(self.d)
        
        # Sample endpoints
        x0 = Poisson(lam0).sample()
        x1 = Poisson(lam1).sample()
        
        # Conditioning includes both λ values
        z = torch.cat([lam0, lam1], dim=0)
        
        return x0, x1, z
    
    def get_context_dim(self):
        return self.d * 2  # [lam0, lam1]


class BetaBinomialDataset(BaseCountDataset):
    """
    BetaBinomial Dataset: x0 ~ BetaBinomial(n₀, α₀, β₀), x1 ~ BetaBinomial(n₁, α₁, β₁)
    
    Uses small alpha/beta values to create challenging distributions with high variance.
    
    - If fixed_base=True: (n₀, α₀, β₀) is fixed, (n₁, α₁, β₁) is random
    - If fixed_base=False: All parameters are random
    """
    
    def __init__(self, size, d, n_scale=50, alpha_range=(0.1, 2.0), beta_range=(0.1, 2.0), 
                 fixed_base=False, fixed_n=50, fixed_alpha=0.5, fixed_beta=0.5,
                 batch_size=None, homogeneous=False):
        self.n_scale = n_scale
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.fixed_n = fixed_n
        self.fixed_alpha = fixed_alpha
        self.fixed_beta = fixed_beta
        super().__init__(size, d, fixed_base, batch_size, homogeneous)
    
    def _generate_fixed_base(self):
        """Generate fixed (n₀, α₀, β₀) for all samples"""
        return {
            'n': torch.full((self.d,), self.fixed_n, dtype=torch.float32),
            'alpha': torch.full((self.d,), self.fixed_alpha, dtype=torch.float32),
            'beta': torch.full((self.d,), self.fixed_beta, dtype=torch.float32)
        }
    
    def _sample_bb_params(self):
        """Sample BetaBinomial parameters"""
        n = torch.randint(5, self.n_scale + 1, (self.d,)).float()
        alpha = self.alpha_range[0] + (self.alpha_range[1] - self.alpha_range[0]) * torch.rand(self.d)
        beta = self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) * torch.rand(self.d)
        return n, alpha, beta
    
    def _sample_betabinomial(self, n, alpha, beta):
        """Sample from BetaBinomial distribution"""
        # Sample p from Beta(alpha, beta)
        p = Beta(alpha, beta).sample()
        # Sample x from Binomial(n, p)
        x = Binomial(total_count=n.long(), probs=p).sample()
        return x
    
    def _generate_endpoints(self):
        if self.fixed_base:
            # Fixed parameters for x₀
            n1 = self.base_params['n']
            alpha1 = self.base_params['alpha']
            beta1 = self.base_params['beta']
            # Random parameters for x₁
            n0, alpha0, beta0 = self._sample_bb_params()
        else:
            # Both random
            n0, alpha0, beta0 = self._sample_bb_params()
            n1, alpha1, beta1 = self._sample_bb_params()
        
        # Sample endpoints
        x0 = self._sample_betabinomial(n0, alpha0, beta0)
        x1 = self._sample_betabinomial(n1, alpha1, beta1)
        
        # Conditioning includes all parameters
        z = torch.cat([n0, alpha0, beta0, n1, alpha1, beta1], dim=0)
        
        return x0, x1, z
    
    def get_context_dim(self):
        return self.d * 6  # [n0, alpha0, beta0, n1, alpha1, beta1]


class PoissonBridgeCollate:
    """Collate function for Poisson bridge sampling with time scheduling"""
    
    def __init__(self, n_steps, time_schedule="uniform", **schedule_kwargs):
        """
        Args:
            n_steps: Number of diffusion steps
            time_schedule: Time point distribution
            **schedule_kwargs: Additional schedule parameters
        """
        self.n_steps = n_steps
        
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
        """Apply Poisson bridge sampling to a batch"""
        # Stack batch items
        x0_batch = torch.stack([item['x0'] for item in batch])
        x1_batch = torch.stack([item['x1'] for item in batch]) 
        z_batch = torch.stack([item['z'] for item in batch])
        
        # Get device from input tensors
        device = x0_batch.device
        
        # Sample time for each item in batch
        t_list = []
        t_idx_list = []
        for _ in range(len(batch)):
            t, t_idx = self._sample_time()
            t_list.append(t)
            t_idx_list.append(t_idx)
        
        t_batch = torch.tensor(t_list, dtype=torch.float32, device=device)
        
        # Batch Poisson bridge sampling
        n = (x1_batch - x0_batch).abs().long()
        k = Binomial(total_count=n, probs=t_batch.unsqueeze(-1)).sample().to(device)
        x_t_batch = x0_batch + torch.sign(x1_batch - x0_batch) * k
        
        return {
            'x0': x0_batch,
            'x1': x1_batch, 
            'x_t': x_t_batch,
            't': t_batch,
            'z': z_batch,
            'r': None  # No r parameter for Poisson bridge
        }


class NBBridgeCollate:
    """Collate function for Negative Binomial (Polya) bridge sampling with time scheduling"""
    
    def __init__(self, n_steps, r_min=1.0, r_max=20.0, r_schedule="linear", 
                 time_schedule="uniform", **schedule_kwargs):
        """
        Args:
            n_steps: Number of diffusion steps
            r_min, r_max: Range for r(t) schedule
            r_schedule: Schedule type for r(t)
            time_schedule: Time point distribution
            **schedule_kwargs: Additional schedule parameters
        """
        self.n_steps = n_steps
        
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
        """Apply Negative Binomial bridge sampling to a batch"""
        # Stack batch items
        x0_batch = torch.stack([item['x0'] for item in batch])
        x1_batch = torch.stack([item['x1'] for item in batch]) 
        z_batch = torch.stack([item['z'] for item in batch])
        
        # Get device from input tensors
        device = x0_batch.device
        
        # Sample time for each item in batch
        t_list = []
        t_idx_list = []
        for _ in range(len(batch)):
            t, t_idx = self._sample_time()
            t_list.append(t)
            t_idx_list.append(t_idx)
        
        t_batch = torch.tensor(t_list, dtype=torch.float32, device=device)
        
        # Get Φ(t) values for each sample in batch (proper bridge parameters)
        phi_t_batch = torch.tensor([self.phi_sched[t_idx].item() for t_idx in t_idx_list], 
                                 dtype=torch.float32, device=device)
        
        # Batch Beta-Binomial bridge sampling using proper formulation
        n = (x1_batch - x0_batch).abs().long()
        
        # Proper alpha/beta computation: α_t = Φ(t), β_t = R - Φ(t)
        alpha = torch.clamp(phi_t_batch.unsqueeze(-1), min=1e-3)
        beta = torch.clamp((self.R - phi_t_batch).unsqueeze(-1), min=1e-3)
        
        # Sample p from Beta distribution
        p = Beta(alpha, beta).sample().clamp(1e-6, 1-1e-6).to(device)
        
        # Sample k from Binomial
        k = Binomial(total_count=n, probs=p).sample().to(device)
        x_t_batch = x0_batch + torch.sign(x1_batch - x0_batch) * k
        
        return {
            'x0': x0_batch,
            'x1': x1_batch,
            'x_t': x_t_batch, 
            't': t_batch,
            'z': z_batch,
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
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
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
        x0 = torch.stack([b["x0"] for b in batch])  # (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
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
        k_idx = torch.randint(1, self.n_steps, (B,), device=device)
        t     = grid[k_idx]                               # (B,)
        w_t   = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1          # (B,)
        N_t   = torch.round(N.float() * w_t.unsqueeze(-1)).long()       # (B, d)

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
        x0 = torch.stack([b["x0"] for b in batch])  # (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
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
        k_idx = torch.randint(1, self.n_steps, (B,), device=device)
        t     = grid[k_idx]                               # (B,)
        w_t   = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1          # (B,)
        N_t   = torch.round(N.float() * w_t.unsqueeze(-1)).long()       # (B, d)

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
        **schedule_kwargs,
    ):
        """
        Args:
          n_steps: number of grid points in [0,1]
          lam0, lam1: birth=death rates at t=0 and t=1
          time_schedule: "uniform" | "linear" | …
        """
        self.n_steps = n_steps

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
        return torch.randint(1, self.n_steps - 1, size=(batch_size,))

    def __call__(self, batch):
        """
        Input:
          batch: list of dicts with keys 'x0','x1','z'.
                 Each 'x0','x1' is a Tensor shape (d,), integer counts.
                 'z' is context (ignored).
        Output dict:
          x0:   (B,d)
          x1:   (B,d)
          x_t:  (B,d)
          t:    (B,)   sampled time indices in (0,1)
          N:    (B,d) latent total jumps at t=1
          B1:   (B,d) births at t=1 in the signed parent
          N_t:  (B,d) total jumps at time t
          B_t:  (B,d) births at time t in the signed parent
          z:    (B,d) context
        """
        x0 = torch.stack([item['x0'] for item in batch], dim=0)  # (B,d)
        x1 = torch.stack([item['x1'] for item in batch], dim=0)  # (B,d)
        z  = torch.stack([item.get('z', torch.zeros_like(x0[0])) for item in batch], dim=0)
        device = x0.device
        B, d = x0.shape

        # 1) pick random interior time‐index k
        k_idxs = self._sample_time_idx(B).to(device)  # (B,)
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


def create_dataset(dataset_type, **kwargs):
    """
    Factory function to create dataset instances
    
    Args:
        dataset_type: "poisson" or "betabinomial"
        **kwargs: Arguments passed to dataset constructor
    
    Returns:
        Dataset instance
    """
    if dataset_type == "poisson":
        return PoissonDataset(**kwargs)
    elif dataset_type == "betabinomial":
        return BetaBinomialDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_dataloader(bridge_type, dataset_type, batch_size, **kwargs):
    """
    Factory function to create DataLoader with appropriate collate function
    
    Args:
        bridge_type: "poisson", "nb", "poisson_bd", "polya_bd", or "reflected_bd"
        dataset_type: "poisson" or "betabinomial"
        batch_size: Batch size for DataLoader
        **kwargs: Arguments passed to Dataset and Collate constructors
    
    Returns:
        DataLoader instance with custom collate function, Dataset instance
    """
    # Extract collate-specific arguments
    collate_kwargs = {}
    dataset_kwargs = {}
    
    # Split kwargs between dataset and collate
    dataset_keys = ['size', 'd', 'lam_scale', 'fixed_base', 'fixed_lam', 
                   'n_scale', 'alpha_range', 'beta_range', 'fixed_n', 'fixed_alpha', 'fixed_beta']
    collate_keys = ['n_steps', 'r_min', 'r_max', 'r_schedule', 'time_schedule',
                   'decay_rate', 'steepness', 'midpoint', 'power', 'concentration',
                   'r', 'beta', 'lam_p0', 'lam_p1', 'lam_m0', 'lam_m1', 'schedule_type',
                   'lam0', 'lam1']  # Added lam0, lam1 for reflected_bd
    
    for key, value in kwargs.items():
        if key in dataset_keys:
            dataset_kwargs[key] = value
        elif key in collate_keys:
            collate_kwargs[key] = value
    
    # Create dataset
    dataset = create_dataset(dataset_type, **dataset_kwargs)
    
    # Create appropriate collate function
    if bridge_type == "poisson":
        collate_fn = PoissonBridgeCollate(**collate_kwargs)
    elif bridge_type == "nb":
        collate_fn = NBBridgeCollate(**collate_kwargs)
    elif bridge_type == "poisson_bd":
        collate_fn = PoissonBDBridgeCollate(**collate_kwargs)
    elif bridge_type == "polya_bd":
        collate_fn = PolyaBDBridgeCollate(**collate_kwargs)
    elif bridge_type == "reflected_bd":
        collate_fn = ReflectedBDBridgeCollate(**collate_kwargs)
    else:
        raise ValueError(f"Unknown bridge type: {bridge_type}")
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, dataset


class InfiniteDataLoader:
    """Wrapper to make DataLoader infinite for training"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            if self.iterator is None:
                self.iterator = iter(self.dataloader)
            return next(self.iterator)
        except StopIteration:
            # Reset iterator when epoch ends
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
    
    def __len__(self):
        return len(self.dataloader)
 