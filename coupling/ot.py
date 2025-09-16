"""
Implement an OT coupling class that implements a call function so it can be used as a collate function for a DataLoader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate
from typing import Optional, Callable
import warnings

import ot

import numpy as np

ot_fns = {
    'sinkhorn': ot.sinkhorn,
    'emd': ot.emd,
    'bregman': ot.bregman.empirical_sinkhorn
}


class OTCollate(nn.Module):
    def __init__(
        self, 
        ot_type: str = 'sinkhorn',
        ot_params: Optional[dict] = None,
        pairwise_dist_fn: Optional[Callable] = None, 
        replace: bool = True,
        scale: float = 1.0
    ):
        super(OTCollate, self).__init__()
        if ot_params is None:
            ot_params = {}
        self.ot_type = ot_type
        self.ot_params = ot_params
        self.replace = replace
        self.scale = scale

        if pairwise_dist_fn is None:
            self.pairwise_dist_fn = lambda x, y: torch.cdist(x, y, p=2) ** 2
        else:
            self.pairwise_dist_fn = pairwise_dist_fn

    def __call__(self, batch, ot_type: Optional[str] = None, ot_params: Optional[dict] = None):
        if ot_type is None:
            ot_type = self.ot_type
        if ot_params is None:
            ot_params = self.ot_params
        
        batch = default_collate(batch)
        
        x_source = batch['x_1']  # Shape: (set_size, dim)
        x_target = batch['x_0']  # Shape: (set_size, dim)
        
        set_size = x_source.shape[0]
        
        # Flatten if needed (for multi-dimensional features)
        if x_source.dim() > 2:
            x_source_flat = x_source.reshape(x_source.shape[0], -1)
        else:
            x_source_flat = x_source
        if x_target.dim() > 2:
            x_target_flat = x_target.reshape(x_target.shape[0], -1)
        else:
            x_target_flat = x_target

        # Use squared Euclidean distance for cost matrix
        cost = self.pairwise_dist_fn(x_source_flat.float() / self.scale, x_target_flat.float() / self.scale)
        
        # Create uniform marginal distributions
        a = ot.unif(set_size)
        b = ot.unif(set_size)
        
        # Convert cost matrix to numpy for OT library
        cost_np = cost.detach().cpu().numpy()
        
        # Compute OT plan
        G = ot_fns[ot_type](a, b, cost_np, **ot_params)
        
        # Check for numerical issues
        if not np.all(np.isfinite(G)):
            warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            G = np.ones_like(G) / G.size
        if np.abs(G.sum()) < 1e-8:
            warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            G = np.ones_like(G) / G.size
        
        # Sample from the OT plan
        G_flat = G.flatten()
        G_flat = G_flat / G_flat.sum()  # Normalize to ensure it's a valid probability distribution
        
        choices = np.random.choice(
            G.shape[0] * G.shape[1], 
            p=G_flat, 
            size=set_size, 
            replace=self.replace
        )
        idx0, idx1 = np.divmod(choices, G.shape[1])
        
        # Sample according to the OT plan
        x_source_sampled = x_source[idx0]  # Shape: (set_size, dim)
        x_target_sampled = x_target[idx1]  # Shape: (set_size, dim)
        
        # Create processed sample
        ot_batch = batch.copy()
        ot_batch['x_1'] = x_source_sampled.int()
        ot_batch['x_0'] = x_target_sampled.int()
        
        # Apply default PyTorch collation to batch the processed samples
        return ot_batch