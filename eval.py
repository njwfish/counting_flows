"""
Model Evaluation Functions

Handles sample generation, trajectory computation, and type conversions.
Provides clean numpy arrays to plotting functions.
"""

import torch
import numpy as np
import logging
from typing import Tuple, List, Dict, Any
from bridges.cupy.constrained import SkellamMeanConstrainedBridge

def generate_evaluation_data(model: torch.nn.Module, bridge: Any, dataset: Any, 
                           n_samples: int = 200) -> Dict[str, np.ndarray]:
    """
    Generate comprehensive evaluation data from trained model.
    
    Returns all data as numpy arrays for easy plotting.
    
    Args:
        model: Trained PyTorch model
        bridge: Bridge object (CuPy-based)
        dataset: Dataset to sample from
        n_samples: Number of samples to generate
        
    Returns:
        Dictionary containing:
        - 'x0_generated': Final generated samples [n_samples, d]
        - 'x0_target': Target samples [n_samples, d] 
        - 'x1_batch': Source samples [n_samples, d]
        - 'x_trajectory': List of [n_samples, d] arrays showing x_t evolution
        - 'x_hat_trajectory': List of [n_samples, d] arrays showing x̂₀ predictions  
        - 'M_trajectory': List of [n_samples, d] arrays showing M_t evolution
    """
    logging.info(f"Generating evaluation data with {n_samples} samples...")
    
    # Get sample data from dataset
    x1_batch = []
    x0_target = []
    
    for i in range(n_samples):
        sample = dataset[i % len(dataset)]
        x1_batch.append(sample['x_1'])
        x0_target.append(sample['x_0'])
    
    x1_batch = torch.stack(x1_batch)
    x0_target = torch.stack(x0_target)
    
    # Convert to numpy for further processing
    x1_np = x1_batch.cpu().numpy()
    x0_target_np = x0_target.cpu().numpy()
    
    # Prepare model
    model.cuda()
    model.eval()
    
    # Check if this is a mean constrained bridge
    is_mean_constrained = isinstance(bridge, SkellamMeanConstrainedBridge)
    
    with torch.no_grad():
        # Prepare arguments for reverse sampler
        sampler_kwargs = {
            'x_1': x1_batch.cuda(),
            'z': {},  # No conditioning
            'model': model,
            'return_trajectory': True,
            'return_x_hat': True, 
            'return_M': True
        }
        
        # Add target mean for mean constrained bridges (allowed "cheating")
        if is_mean_constrained:
            mu_0 = x0_target.float().mean(axis=0)
            sampler_kwargs['mu_0'] = mu_0.cuda()
            logging.info("Using mean constrained bridge with target mean")
        
        # Generate samples and trajectories
        result = bridge.reverse_sampler(**sampler_kwargs)
        
        # Parse results and convert from CuPy to numpy
        if isinstance(result, tuple) and len(result) == 4:
            x0_generated, x_trajectory, x_hat_trajectory, M_trajectory = result
            
            # Convert CuPy arrays to numpy
            x0_generated_np = x0_generated.get().astype(np.float32)
            
            # Convert trajectory arrays to lists of numpy arrays
            x_trajectory_np = x_trajectory.get().astype(np.float32)
            x_hat_trajectory_np = x_hat_trajectory.get().astype(np.float32)
            M_trajectory_np = M_trajectory.get().astype(np.float32)
            
            # Ensure trajectories start with x1
            if len(x_trajectory_np) > 0 and not np.allclose(x_trajectory_np[0], x1_np, atol=1e-6):
                x_trajectory_np = [x1_np.astype(np.float32)] + x_trajectory_np
                
    # Package results
    eval_data = {
        'x0_generated': x0_generated_np,
        'x0_target': x0_target_np.astype(np.float32),
        'x1_batch': x1_np.astype(np.float32),
        'x_trajectory': x_trajectory_np,
        'x_hat_trajectory': x_hat_trajectory_np,
        'M_trajectory': M_trajectory_np
    }
    
    logging.info("Evaluation data generation completed")
    return eval_data


def compute_evaluation_metrics(eval_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute evaluation metrics from generated samples.
    
    Args:
        eval_data: Dictionary from generate_evaluation_data()
        
    Returns:
        Dictionary of computed metrics
    """
    x0_generated = eval_data['x0_generated']
    x0_target = eval_data['x0_target']
    
    # Mean errors
    generated_mean = x0_generated.mean(axis=0)
    target_mean = x0_target.mean(axis=0)
    mean_error = np.abs(generated_mean - target_mean).mean()
    
    # Variance errors
    generated_var = x0_generated.var(axis=0)
    target_var = x0_target.var(axis=0)
    var_error = np.abs(generated_var - target_var).mean()
    
    # Per-dimension errors
    per_dim_mean_error = np.abs(generated_mean - target_mean)
    per_dim_var_error = np.abs(generated_var - target_var)
    
    metrics = {
        'mean_absolute_error': mean_error,
        'variance_absolute_error': var_error,
        'max_mean_error': per_dim_mean_error.max(),
        'max_var_error': per_dim_var_error.max(),
        'mean_correlation': np.corrcoef(generated_mean, target_mean)[0, 1] if len(generated_mean) > 1 else 1.0
    }
    
    return metrics


def log_evaluation_summary(eval_data: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    """Log a summary of evaluation results."""
    x0_generated = eval_data['x0_generated']
    x0_target = eval_data['x0_target']
    
    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Generated samples shape: {x0_generated.shape}")
    logging.info(f"Generated mean: {x0_generated.mean(axis=0)}")
    logging.info(f"Target mean:    {x0_target.mean(axis=0)}")
    logging.info(f"Generated std:  {x0_generated.std(axis=0)}")
    logging.info(f"Target std:     {x0_target.std(axis=0)}")
    logging.info("-" * 40)
    logging.info("METRICS:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.4f}")
    logging.info("=" * 60) 