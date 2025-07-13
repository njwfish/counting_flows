"""
Simple PyTorch Datasets for Count-based Flow Matching

Standard PyTorch datasets with mixture of Poissons.
"""

import torch
from torch.utils.data import Dataset
from torch.distributions import Poisson, HalfNormal, Dirichlet
import numpy as np
from typing import Dict, Any, Tuple


class PoissonMixtureDataset(Dataset):
    """
    Mixture of Poissons dataset for count-based flow matching.
    
    Pre-samples all source and target count vectors at initialization
    for fast training with mixtures of Poisson distributions.
    """
    
    def __init__(
        self,
        size: int,
        d: int,
        k: int = 3,
        lambda_scale: float = 10.0,
        resample_every: int = 1000
    ):
        """
        Args:
            size: Dataset size (number of samples)
            d: Dimensionality of count vectors  
            k: Number of mixture components
            lambda_scale: Scale parameter for sampling lambdas
            resample_every: Resample mixture parameters every N samples
        """
        self.size = size
        self.d = d
        self.k = k
        self.lambda_scale = lambda_scale
        self.resample_every = resample_every
        
        print(f"Pre-sampling {size} samples with d={d}, k={k}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization"""
        # Calculate how many parameter sets we need
        num_param_sets = (self.size + self.resample_every - 1) // self.resample_every
        
        # Pre-allocate tensors for all data
        self.x0_data = torch.zeros(self.size, self.d, dtype=torch.int32)
        self.x1_data = torch.zeros(self.size, self.d, dtype=torch.int32)
        
        sample_idx = 0
        
        for param_set in range(num_param_sets):
            # Sample new mixture parameters
            lambda_source, lambda_target, weights_source, weights_target = self._sample_parameters()
            
            # Determine how many samples to generate with these parameters
            samples_this_set = min(self.resample_every, self.size - sample_idx)
            
            # Generate samples for this parameter set
            for _ in range(samples_this_set):
                # Sample from source and target mixtures
                x0 = self._sample_from_mixture(lambda_source, weights_source)
                x1 = self._sample_from_mixture(lambda_target, weights_target)
                
                # Store in pre-allocated tensors
                self.x0_data[sample_idx] = x0
                self.x1_data[sample_idx] = x1
                sample_idx += 1
    
    def _sample_parameters(self):
        """Sample new mixture parameters"""
        # Sample lambda matrices using HalfNormal to ensure positivity
        # Shape: (k, d) for both source and target
        half_normal = HalfNormal(self.lambda_scale)
        lambda_source = half_normal.sample((self.k, self.d))  # [k, d]
        lambda_target = half_normal.sample((self.k, self.d))  # [k, d]
        
        # Sample mixture weights using Dirichlet
        weights_source = Dirichlet(torch.ones(self.k)).sample()  # [k]
        weights_target = Dirichlet(torch.ones(self.k)).sample()  # [k]
        
        return lambda_source, lambda_target, weights_source, weights_target
    
    def _sample_from_mixture(self, lambdas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Sample from mixture of Poissons
        
        Args:
            lambdas: Lambda parameters [k, d]
            weights: Mixture weights [k]
        
        Returns:
            Sample from mixture [d]
        """
        # Sample which component to use
        component = torch.multinomial(weights, 1).item()
        
        # Sample from that component's Poisson
        sample = Poisson(lambdas[component]).sample()  # [d]
        
        return sample.int()
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[idx]
        }
 