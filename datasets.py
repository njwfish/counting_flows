"""
Simple PyTorch Datasets for Count-based Flow Matching

Standard PyTorch datasets with mixture of Poissons.
"""

import torch
from torch.utils.data import Dataset
from torch.distributions import Poisson, Normal, Dirichlet
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
        lambda_loc: float = 20.0,
        lambda_scale: float = 20.0,
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
        self.lambda_loc = lambda_loc
        
        print(f"Pre-sampling {size} samples with d={d}, k={k}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization using vectorized operations"""
        # Calculate how many parameter sets we need        
        # Pre-allocate tensors for all data
        self.x0_data = torch.zeros(self.size, self.d, dtype=torch.int32)
        self.x1_data = torch.zeros(self.size, self.d, dtype=torch.int32)
        
        sample_idx = 0
        
        # Sample new mixture parameters
        lambda_source, lambda_target, weights_source, weights_target = self._sample_parameters()
        
        # Generate all samples for this parameter set vectorized
        self.x0_data = self._sample_from_mixture_vectorized(lambda_source, weights_source, self.size)
        self.x1_data = self._sample_from_mixture_vectorized(lambda_target, weights_target, self.size)
 
    
    def _sample_parameters(self):
        """Sample new mixture parameters"""
        # Sample lambda matrices using Normal to ensure positivity
        # Shape: (k, d) for both source and target
        normal = Normal(scale=self.lambda_scale, loc=self.lambda_loc)
        lambda_source = torch.abs(normal.sample((self.k, self.d)))  # [k, d]
        lambda_target = torch.abs(normal.sample((self.k, self.d)))  # [k, d]
        
        # Sample mixture weights using Dirichlet
        weights_source = Dirichlet(torch.ones(self.k)).sample()  # [k]
        weights_target = Dirichlet(torch.ones(self.k)).sample()  # [k]
        
        return lambda_source, lambda_target, weights_source, weights_target
    
    def _sample_from_mixture_vectorized(self, lambdas: torch.Tensor, weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Vectorized sampling from mixture of Poissons
        
        Args:
            lambdas: Lambda parameters [k, d]
            weights: Mixture weights [k]
            num_samples: Number of samples to generate
        
        Returns:
            Batch of samples from mixture [num_samples, d]
        """
        # Sample which components to use for each sample - vectorized
        # Shape: [num_samples]
        components = torch.multinomial(weights, num_samples, replacement=True)
        
        # Use advanced indexing to select lambda parameters for each sample
        # Shape: [num_samples, d]
        selected_lambdas = lambdas[components]  # Broadcasting: [num_samples] -> [num_samples, d]
        
        # Sample from Poisson distributions vectorized
        # Each row of selected_lambdas contains the lambda parameters for one sample
        samples = Poisson(selected_lambdas).sample()  # [num_samples, d]
        
        return samples.int()
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        target_idx = np.random.randint(0, self.size)
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[target_idx]
        }
 