"""
Discrete Moons Dataset for Count-based Flow Matching

Integerized version of sklearn make_moons dataset scaled to discrete vocabulary.
"""

import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np
from typing import Dict, Any


class DiscreteMoonsDataset(Dataset):
    """
    Discrete version of 2-moons dataset for discrete flow matching.
    
    Converts continuous 2D moons data to discrete integers in vocab_size range
    following the scaling: round(clip(x * scale + offset, 0, vocab_size-1))
    """
    
    def __init__(
        self,
        size: int,
        vocab_size: int = 128,
        scale: float = 35.0,
        offset: float = 50.0,
        noise: float = 0.05,
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
        self.vocab_size = vocab_size
        self.scale = scale
        self.offset = offset
        self.noise = noise
        
        print(f"Pre-sampling {size} discrete moons samples with vocab_size={vocab_size}...")
        
        # Pre-sample all data
        self._pre_sample_all_data()
        
        print(f"✓ Pre-sampling complete!")
    
    def _pre_sample_all_data(self):
        """Pre-sample all source and target data at initialization"""
        # Generate continuous moons data
        moons_data, _ = make_moons(self.size, noise=self.noise)
        
        # Convert to discrete integers following notebook scaling
        # x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()
        moons_tensor = torch.tensor(moons_data, dtype=torch.float32)
        self.x1_data = torch.round(
            torch.clamp(moons_tensor * self.scale + self.offset, 
                       min=0.0, max=self.vocab_size - 1)
        ).long()
        
        # Generate random noise data for x_0
        # x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, 2))
        self.x0_data = torch.randint(
            low=0, high=self.vocab_size, size=(self.size, 2)
        )
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return pre-computed sample"""
        # For target, use random pairing like in original dataset
        target_idx = np.random.randint(0, self.size)
        return {
            'x_0': self.x0_data[idx],
            'x_1': self.x1_data[target_idx]
        } 