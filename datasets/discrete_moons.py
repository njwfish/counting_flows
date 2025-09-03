"""
Discrete 8-Gaussians to 2-Moons Dataset for Count-based Flow Matching

Integerized version of 8-gaussians -> 2-moons task scaled to discrete vocabulary.
"""

import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np
from typing import Dict, Any


class DiscreteMoonsDataset(Dataset):
    """
    Discrete version of 8-gaussians -> 2-moons dataset for discrete flow matching.
    
    Converts continuous 2D data to discrete integers in vocab_size range
    following the scaling: round(clip(x * scale + offset, 0, vocab_size-1))
    """
    
    def __init__(
        self,
        size: int,
        value_range: int = 128,
        min_value: int = 0,
        scale: float = 35.0,
        offset: float = 50.0,
        noise: float = 0.05,
        data_dim: int = 2,
        flip_moons: bool = False,
    ):
        """
        Args:
            size: Dataset size (number of samples)
            vocab_size: Size of discrete vocabulary (0 to vocab_size-1)
            scale: Scaling factor for continuous data
            offset: Offset for continuous data before discretization
            noise: Noise level for make_moons
        """
        self.size = size
        self.value_range = value_range
        self.min_value = min_value
        self.scale = scale
        self.offset = offset
        self.noise = noise
        self.data_dim = data_dim
        self.flip_moons = flip_moons
        
        print(f"Pre-sampling {size} discrete moons samples with range={value_range}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _make_8_gaussians(self, n_samples, noise: float = 0.02):
        """Generate 8 gaussians arranged in a circle"""
        n_gaussians = 8
        variance = noise
        radius = 2.0
        
        # Generate angles for 8 gaussians evenly spaced around circle
        angles = np.linspace(0, 2*np.pi, n_gaussians, endpoint=False)
        
        # Centers of gaussians
        centers = np.array([(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles])
        
        # Sample from each gaussian
        samples = []
        for i in range(n_samples):
            # Choose a random gaussian
            gaussian_idx = np.random.randint(n_gaussians)
            center = centers[gaussian_idx]
            
            # Sample from that gaussian
            sample = np.random.normal(center, variance)
            samples.append(sample)
        
        return np.array(samples)
    
    def _discretize_data(self, continuous_data):
        """Convert continuous data to discrete integers using scaling"""
        data_tensor = torch.tensor(continuous_data, dtype=torch.float32)
        discrete_data = torch.round(
            torch.clamp(data_tensor * self.scale + self.offset, 
                       min=self.min_value, max=self.value_range - 1)
        ).int()
        return discrete_data
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization"""
        # Generate 8 gaussians data for x_0 (source)
        gaussians_data = self._make_8_gaussians(self.size, noise=self.noise)
        self.x1_data = self._discretize_data(gaussians_data)
        
        # Generate 2 moons data for x_1 (target)
        moons_data, _ = make_moons(self.size, noise=self.noise)
        moons_data = moons_data - np.array([0.5, 0.25])
        self.x0_data = self._discretize_data(moons_data)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        # For target, use random pairing like in original dataset
        target_idx = np.random.randint(0, self.size)
        if self.flip_moons:
            return {
                'x_0': self.x1_data[idx],
                'x_1': self.x0_data[target_idx]
            } 
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[target_idx]
        } 