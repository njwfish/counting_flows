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
from .scheduling import make_r_schedule, make_time_spacing_schedule
from abc import ABC, abstractmethod


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
        
        # Create r(t) schedule
        self.r_sched = make_r_schedule(n_steps, r_min, r_max, r_schedule, **schedule_kwargs)
        
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
        
        # Get r(t) values for each sample in batch
        r_t_batch = torch.tensor([self.r_sched[t_idx].item() for t_idx in t_idx_list], 
                                dtype=torch.float32, device=device)
        
        # Batch Beta-Binomial bridge sampling
        n = (x1_batch - x0_batch).abs().long()
        
        # Vectorized alpha/beta computation
        alpha = torch.clamp(r_t_batch.unsqueeze(-1) * t_batch.unsqueeze(-1), min=1e-3)
        beta = torch.clamp(r_t_batch.unsqueeze(-1) * (1 - t_batch.unsqueeze(-1)), min=1e-3)
        
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
            'r': r_t_batch
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
        bridge_type: "poisson" or "nb"
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
                   'decay_rate', 'steepness', 'midpoint', 'power', 'concentration']
    
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
