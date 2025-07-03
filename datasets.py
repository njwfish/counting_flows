"""
PyTorch Datasets for Count-based Flow Matching

Provides two core dataset classes for count flow scenarios:
1. PoissonDataset: Poisson endpoints with configurable base measure
2. BetaBinomialDataset: BetaBinomial endpoints with small alpha/beta (tricky distribution)

Each dataset can use either fixed or random base measures.
Time scheduling is handled by bridge classes, not datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Poisson, Binomial, Beta
import numpy as np
from counting_flows.bridges.numpy.skellam import SkellamBridge
from counting_flows.bridges.numpy.constrained import SkellamMeanConstrainedBridge
from abc import ABC, abstractmethod


class BaseCountDataset(Dataset, ABC):
    """Base class for count flow datasets"""
    
    def __init__(self, size, d, fixed_base=False, batch_size=None, homogeneous=True):
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
        return self._generate_batch(self.batch_size, self.homogeneous)

    
    def _generate_batch(self, batch_size, homogeneous=True):
        """Generate a batch of endpoint pairs"""
        if homogeneous:
            # Generate one set of parameters and use them to sample multiple different endpoints
            z = self._sample_parameters()
            
            x0_list, x1_list = [], []
            for _ in range(batch_size):
                x0, x1 = self._sample_from_parameters(z)
                x0_list.append(x0)
                x1_list.append(x1)
            
            x0_batch = torch.stack(x0_list)
            x1_batch = torch.stack(x1_list)
            z_batch = z.unsqueeze(0).repeat(batch_size, 1)
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
    def _generate_fixed_base(self):
        """Generate fixed base measure parameters"""
        pass
    
    @abstractmethod
    def get_context_dim(self):
        """Return context dimension for this dataset"""
        pass
    
    @abstractmethod
    def _sample_parameters(self):
        """Sample conditioning parameters z for homogeneous batches"""
        pass
    
    @abstractmethod
    def _sample_from_parameters(self, z):
        """Sample x0, x1 from given parameters z"""
        pass


class PoissonDataset(BaseCountDataset):
    """
    Poisson Dataset: x0 ~ Poisson(λ₀), x1 ~ Poisson(λ₁)
    
    - If fixed_base=True: λ₀ is fixed, λ₁ is random
    - If fixed_base=False: Both λ₀ and λ₁ are random
    """
    
    def __init__(self, size, d, lam_scale=50.0, fixed_base=False, fixed_lam=10.0, 
                 batch_size=None, homogeneous=True):
        self.lam_scale = lam_scale
        self.fixed_lam = fixed_lam
        super().__init__(size, d, fixed_base, batch_size, homogeneous)
        
    def _generate_fixed_base(self):
        """Generate fixed λ₀ for all samples"""
        return torch.full((self.d,), self.fixed_lam, dtype=torch.float32)
    
    def get_context_dim(self):
        return self.d * 2  # [lam0, lam1]
    
    def _sample_parameters(self):
        """Sample lambda parameters for homogeneous batches"""
        if self.fixed_base:
            # Fixed λ₀, random λ₁
            lam1 = self.base_params
            lam0 = self.lam_scale * torch.rand(self.d)
        else:
            # Both random
            lam0 = self.lam_scale * torch.rand(self.d)
            lam1 = self.lam_scale * torch.rand(self.d)
        
        # Return conditioning parameters
        return torch.cat([lam0, lam1], dim=0)
    
    def _sample_from_parameters(self, z):
        """Sample x0, x1 from given lambda parameters"""
        # Split z back into lam0 and lam1
        lam0 = z[:self.d]
        lam1 = z[self.d:]
        
        # Sample endpoints using these fixed parameters
        x0 = Poisson(lam0).sample()
        x1 = Poisson(lam1).sample()
        
        return x0, x1


class BetaBinomialDataset(BaseCountDataset):
    """
    BetaBinomial Dataset: x0 ~ BetaBinomial(n₀, α₀, β₀), x1 ~ BetaBinomial(n₁, α₁, β₁)
    
    Uses small alpha/beta values to create challenging distributions with high variance.
    
    - If fixed_base=True: (n₀, α₀, β₀) is fixed, (n₁, α₁, β₁) is random
    - If fixed_base=False: All parameters are random
    """
    
    def __init__(self, size, d, n_scale=50, alpha_range=(0.1, 2.0), beta_range=(0.1, 2.0), 
                 fixed_base=False, fixed_n=50, fixed_alpha=0.5, fixed_beta=0.5,
                 batch_size=None, homogeneous=True):
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
    
    
    def get_context_dim(self):
        return self.d * 6  # [n0, alpha0, beta0, n1, alpha1, beta1]
    
    def _sample_parameters(self):
        """Sample BetaBinomial parameters for homogeneous batches"""
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
        
        # Return conditioning parameters
        return torch.cat([n0, alpha0, beta0, n1, alpha1, beta1], dim=0)
    
    def _sample_from_parameters(self, z):
        """Sample x0, x1 from given BetaBinomial parameters"""
        # Split z back into parameter components
        n0 = z[:self.d]
        alpha0 = z[self.d:2*self.d]
        beta0 = z[2*self.d:3*self.d]
        n1 = z[3*self.d:4*self.d]
        alpha1 = z[4*self.d:5*self.d]
        beta1 = z[5*self.d:6*self.d]
        
        # Sample endpoints using these fixed parameters
        x0 = self._sample_betabinomial(n0, alpha0, beta0)
        x1 = self._sample_betabinomial(n1, alpha1, beta1)
        
        return x0, x1


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
    Factory function to create DataLoader with appropriate bridge function
    
    Args:
        bridge_type: "poisson_bd", "reflected_bd", or "poisson_bd_mean"
        dataset_type: "poisson" or "betabinomial"
        batch_size: Batch size for DataLoader
        **kwargs: Arguments passed to Dataset and Bridge constructors
    
    Returns:
        DataLoader instance with custom bridge function, Dataset instance
    """
    # Extract bridge-specific arguments
    bridge_kwargs = {}
    dataset_kwargs = {}
    
    # Split kwargs between dataset and bridge
    dataset_keys = ['size', 'd', 'lam_scale', 'fixed_base', 'fixed_lam', 
                   'n_scale', 'alpha_range', 'beta_range', 'fixed_n', 'fixed_alpha', 'fixed_beta',
                   'homogeneous']  # homogeneous goes to dataset
    bridge_keys = ['n_steps', 'lam0', 'lam1', 'schedule_type', 'time_schedule',
                   'decay_rate', 'steepness', 'midpoint', 'power', 'concentration',
                   'homogeneous_time', 'mh_sweeps']  # homogeneous_time goes to bridges
    
    for key, value in kwargs.items():
        if key in dataset_keys:
            dataset_kwargs[key] = value
        elif key in bridge_keys:
            bridge_kwargs[key] = value
    
    # Always use dataset's built-in batch generation
    dataset_kwargs['batch_size'] = batch_size
    dataset = create_dataset(dataset_type, **dataset_kwargs)
    
    # Create appropriate bridge function
    if bridge_type == "poisson_bd":
        bridge_fn = SkellamBridge(**bridge_kwargs)
    elif bridge_type == "reflected_bd":
        bridge_fn = ReflectedSkellamBridge(**bridge_kwargs)
    elif bridge_type == "poisson_bd_mean":
        bridge_fn = SkellamMeanConstrainedBridge(**bridge_kwargs)
    else:
        raise ValueError(f"Unknown bridge type: {bridge_type}. Supported types: poisson_bd, reflected_bd, poisson_bd_mean")
    
    # Create DataLoader with batch_size=1 since dataset produces full batches
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=bridge_fn,
        num_workers=0,
        pin_memory=False,  # Disabled to avoid GPU tensor pinning issues
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
 