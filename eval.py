"""
Simple evaluation functions for counting flows
"""

import torch
import numpy as np
import logging
from typing import Dict, Any
from bridges.cupy.constrained import SkellamMeanConstrainedBridge


def generate_evaluation_data(model: torch.nn.Module, bridge: Any, dataset: Any, 
                           n_samples: int = 200) -> Dict[str, np.ndarray]:
    """Generate evaluation data from trained model"""
    model.eval()
    
    # Get data efficiently using DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    batch = next(iter(dataloader))
    
    x0_target = batch['x_0'].numpy()
    x1_source = batch['x_1'].numpy()
    
    # Convert to torch tensors only for bridge call
    x0_torch = torch.tensor(x0_target).cuda()
    x1_torch = torch.tensor(x1_source).cuda()
    
    # Check if this is a mean constrained bridge
    is_mean_constrained = isinstance(bridge, SkellamMeanConstrainedBridge)
    
    with torch.no_grad():
        # Prepare arguments for reverse sampler
        sampler_kwargs = {
            'x_1': x1_torch,
            'z': {},
            'model': model.to('cuda'),
            'return_trajectory': True,
            'return_x_hat': True,
            'return_M': True
        }
        
        # Add target mean for mean constrained bridges
        if is_mean_constrained:
            mu_0 = x0_torch.float().mean(dim=0)
            sampler_kwargs['mu_0'] = mu_0
            logging.info("Using mean constrained bridge with target mean")
        
        # Generate samples with trajectories
        result = bridge.reverse_sampler(**sampler_kwargs)
        
        if isinstance(result, tuple) and len(result) == 4:
            x0_generated, x_trajectory, x_hat_trajectory, M_trajectory = result
            
            # Convert CuPy arrays to numpy
            x0_generated = x0_generated.get()
            x_trajectory = x_trajectory.get()
            x_hat_trajectory = x_hat_trajectory.get()
            M_trajectory = M_trajectory.get()
            
        else:
            # Fallback
            x0_generated = result.get()
            x_trajectory = np.array([x1_source, x0_generated])
            x_hat_trajectory = np.array([x0_generated])
            M_trajectory = np.zeros_like(x_trajectory)
    
    return {
        'x0_target': x0_target,
        'x1_batch': x1_source,
        'x0_generated': x0_generated,
        'x_trajectory': x_trajectory,
        'x_hat_trajectory': x_hat_trajectory,
        'M_trajectory': M_trajectory
    }


def compute_evaluation_metrics(eval_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute simple evaluation metrics"""
    x0_generated = eval_data['x0_generated']
    x0_target = eval_data['x0_target']
    
    # Basic statistics using numpy
    gen_mean = np.mean(x0_generated, axis=0)
    target_mean = np.mean(x0_target, axis=0)
    mean_error = np.mean(np.abs(gen_mean - target_mean))
    
    gen_var = np.var(x0_generated, axis=0)
    target_var = np.var(x0_target, axis=0)
    var_error = np.mean(np.abs(gen_var - target_var))
    
    return {
        'mean_error': float(mean_error),
        'var_error': float(var_error),
        'target_mean': target_mean.tolist(),
        'gen_mean': gen_mean.tolist()
    }


def log_evaluation_summary(eval_data: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    """Log evaluation summary"""
    logging.info("=== Evaluation Summary ===")
    logging.info(f"Mean error: {metrics['mean_error']:.4f}")
    logging.info(f"Variance error: {metrics['var_error']:.4f}")
    logging.info(f"Generated {len(eval_data['x0_generated'])} samples") 