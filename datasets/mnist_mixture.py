"""
MNIST + Gaussian Mixture Datasets for Multimodal Training

These datasets combine MNIST images with Gaussian mixture count vectors,
creating multimodal data for testing the multimodal energy score loss.
"""

import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal, Normal, Dirichlet
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .gaussian_mixture import GaussianMixtureDataset, LowRankGaussianMixtureDataset
from .mnist import DiffMNIST


class MNISTGaussianMixtureDataset(Dataset):
    """
    Dataset that combines MNIST images with Gaussian mixture count vectors.
    
    This creates multimodal data where each sample contains:
    - An MNIST image selected based on mixture component assignment
    - A count vector sampled from a Gaussian mixture
    
    Structure:
      x_0: Dict with 'img' (MNIST image) and 'counts' (Gaussian sample)
      x_1: Dict with 'img' (noise) and 'counts' (different Gaussian sample)
    """
    
    def __init__(
        self,
        size: int,
        data_dim: int,
        value_range: int = 1024,
        min_value: int = 0,
        k: int = 5,  # Number of mixture components (should be <= 10 for MNIST digits)
        seed: int = 42,
        **gaussian_kwargs  # Additional args for GaussianMixtureDataset
    ):
        """
        Args:
            size: Dataset size (number of samples)
            data_dim: Dimensionality of count vectors
            value_range: Maximum value for integer count outputs
            min_value: Minimum value for integer count outputs
            k: Number of mixture components (maps to MNIST digits 0 to k-1)
            seed: Random seed for reproducibility
            **gaussian_kwargs: Additional arguments for GaussianMixtureDataset
        """
        super().__init__()
        self.size = size
        self.data_dim = data_dim
        self.value_range = value_range
        self.min_value = min_value
        self.k = min(k, 10)  # Cap at 10 MNIST digits
        self.seed = seed
        
        # Load MNIST dataset
        self.mnist_dataset = DiffMNIST()
        
        # Create Gaussian mixture dataset for counts
        self.gaussian_dataset = GaussianMixtureDataset(
            size=size,
            data_dim=data_dim,
            value_range=value_range,
            min_value=min_value,
            k=self.k,
            seed=seed,
            **gaussian_kwargs
        )
        
        # Group MNIST data by digit labels
        self._group_mnist_by_digit()
        
        # Pre-compute MNIST image selections based on Gaussian component assignments
        self._pre_compute_mnist_selections()
    
    def _group_mnist_by_digit(self):
        """Group MNIST images by digit labels (0-9)"""
        self.mnist_by_digit = {}
        for digit in range(10):
            digit_indices = (self.mnist_dataset.labels == digit).nonzero(as_tuple=True)[0]
            self.mnist_by_digit[digit] = digit_indices
    
    def _pre_compute_mnist_selections(self):
        """Pre-compute MNIST image selections based on Gaussian component assignments"""
        with torch.random.fork_rng():
            torch.manual_seed(self.seed + 1000)  # Different seed for MNIST selection
            
            # Pre-allocate storage for MNIST indices
            self.mnist_indices_x0 = torch.zeros(self.size, dtype=torch.long)
            self.mnist_indices_x1 = torch.zeros(self.size, dtype=torch.long)
            
            for idx in range(self.size):
                # Get component assignments from Gaussian dataset
                component_x0 = self.gaussian_dataset.x0_components[idx].item()
                component_x1 = self.gaussian_dataset.x1_components[idx].item()
                
                # Map component to MNIST digit (use modulo for safety)
                digit_x0 = component_x0 % 10
                digit_x1 = component_x1 % 10
                
                # Select random MNIST images of the appropriate digits
                # Note: MNIST always has images for all digits 0-9, so no fallback needed
                self.mnist_indices_x0[idx] = self.mnist_by_digit[digit_x0][
                    torch.randint(0, len(self.mnist_by_digit[digit_x0]), (1,))
                ]
                
                self.mnist_indices_x1[idx] = self.mnist_by_digit[digit_x1][
                    torch.randint(0, len(self.mnist_by_digit[digit_x1]), (1,))
                ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get Gaussian mixture data
        gaussian_data = self.gaussian_dataset[idx]
        
        # Get MNIST images
        mnist_img_x0 = self.mnist_dataset.data[self.mnist_indices_x0[idx]]  # [1, 28, 28]
        mnist_img_x1 = self.mnist_dataset.data[self.mnist_indices_x1[idx]]  # [1, 28, 28]
        
        # Generate noise for x_1 images (alternative to selected MNIST)
        mnist_noise = torch.randn_like(mnist_img_x0)
        
        return {
            'x_0': {
                'img': mnist_img_x0.float(),           # [1, 28, 28]
                'counts': gaussian_data['x_0']         # [data_dim]
            },
            'x_1': {
                'img': mnist_noise.float(),            # [1, 28, 28] - noise instead of MNIST
                'counts': gaussian_data['x_1']         # [data_dim]
            }
        }


class MNISTLowRankGaussianMixtureDataset(Dataset):
    """
    Dataset that combines MNIST images with low-rank Gaussian mixture count vectors.
    
    Similar to MNISTGaussianMixtureDataset but uses LowRankGaussianMixtureDataset
    for the count component.
    """
    
    def __init__(
        self,
        size: int,
        data_dim: int,
        latent_dim: int,
        value_range: int = 1024,
        min_value: int = 0,
        k: int = 5,  # Number of mixture components
        seed: int = 42,
        **gaussian_kwargs  # Additional args for LowRankGaussianMixtureDataset
    ):
        """
        Args:
            size: Dataset size (number of samples)
            data_dim: Output dimensionality of count vectors
            latent_dim: Latent dimensionality for low-rank structure
            value_range: Maximum value for integer count outputs
            min_value: Minimum value for integer count outputs
            k: Number of mixture components (maps to MNIST digits 0 to k-1)
            seed: Random seed for reproducibility
            **gaussian_kwargs: Additional arguments for LowRankGaussianMixtureDataset
        """
        super().__init__()
        self.size = size
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.value_range = value_range
        self.min_value = min_value
        self.k = min(k, 10)  # Cap at 10 MNIST digits
        self.seed = seed
        
        # Load MNIST dataset
        self.mnist_dataset = DiffMNIST()
        
        # Create low-rank Gaussian mixture dataset for counts
        self.gaussian_dataset = LowRankGaussianMixtureDataset(
            size=size,
            data_dim=data_dim,
            latent_dim=latent_dim,
            value_range=value_range,
            min_value=min_value,
            k=self.k,
            seed=seed,
            **gaussian_kwargs
        )
        
        # Group MNIST data by digit labels
        self._group_mnist_by_digit()
        
        # Pre-compute MNIST image selections based on Gaussian component assignments
        self._pre_compute_mnist_selections()
    
    def _group_mnist_by_digit(self):
        """Group MNIST images by digit labels (0-9)"""
        self.mnist_by_digit = {}
        for digit in range(10):
            digit_indices = (self.mnist_dataset.labels == digit).nonzero(as_tuple=True)[0]
            self.mnist_by_digit[digit] = digit_indices
    
    def _pre_compute_mnist_selections(self):
        """Pre-compute MNIST image selections based on Gaussian component assignments"""
        with torch.random.fork_rng():
            torch.manual_seed(self.seed + 1000)  # Different seed for MNIST selection
            
            # Pre-allocate storage for MNIST indices
            self.mnist_indices_x0 = torch.zeros(self.size, dtype=torch.long)
            self.mnist_indices_x1 = torch.zeros(self.size, dtype=torch.long)
            
            for idx in range(self.size):
                # Get component assignments from Gaussian dataset
                component_x0 = self.gaussian_dataset.x0_components[idx].item()
                component_x1 = self.gaussian_dataset.x1_components[idx].item()
                
                # Map component to MNIST digit (use modulo for safety)
                digit_x0 = component_x0 % 10
                digit_x1 = component_x1 % 10
                
                # Select random MNIST images of the appropriate digits
                # Note: MNIST always has images for all digits 0-9, so no fallback needed
                self.mnist_indices_x0[idx] = self.mnist_by_digit[digit_x0][
                    torch.randint(0, len(self.mnist_by_digit[digit_x0]), (1,))
                ]
                
                self.mnist_indices_x1[idx] = self.mnist_by_digit[digit_x1][
                    torch.randint(0, len(self.mnist_by_digit[digit_x1]), (1,))
                ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get Gaussian mixture data
        gaussian_data = self.gaussian_dataset[idx]
        
        # Get MNIST images
        mnist_img_x0 = self.mnist_dataset.data[self.mnist_indices_x0[idx]]  # [1, 28, 28]
        mnist_img_x1 = self.mnist_dataset.data[self.mnist_indices_x1[idx]]  # [1, 28, 28]
        
        # Generate noise for x_1 images (alternative to selected MNIST)
        mnist_noise = torch.randn_like(mnist_img_x0)
        
        return {
            'x_0': {
                'img': mnist_img_x0.float(),           # [1, 28, 28]
                'counts': gaussian_data['x_0']         # [data_dim]
            },
            'x_1': {
                'img': mnist_noise.float(),            # [1, 28, 28] - noise instead of MNIST
                'counts': gaussian_data['x_1']         # [data_dim]
            }
        }
