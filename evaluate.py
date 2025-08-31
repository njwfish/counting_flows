"""
Evaluation functions for counting flows with smart data type detection and caching
"""

import torch
import numpy as np
import logging
import pickle
import yaml
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable, Union, Optional
from bridges.cupy.constrained import SkellamMeanConstrainedBridge
from bridges.cupy.skellam import SkellamBridge
from typing import List

def detect_data_type(dataset: Any) -> str:
    """Detect if dataset contains vector, image, or multimodal data"""
    try:
        sample = dataset[0]
        
        # Check x_0 shape to determine data type
        x_0 = sample['x_0']
        
        if len(x_0.shape) == 1:
            return 'vector'
        elif len(x_0.shape) == 3:  # [C, H, W] or [H, W, C]
            return 'image'
        elif len(x_0.shape) == 2:
            # Could be [batch, features] or [H, W] - check size
            if x_0.shape[0] > 1 and x_0.shape[1] > 1 and max(x_0.shape) > 50:
                return 'image'  # Likely image data
            else:
                return 'vector'  # Likely vector features
        else:
            return 'multimodal'  # Complex shape, handle specially
            
    except Exception as e:
        logging.warning(f"Could not detect data type: {e}, defaulting to vector")
        return 'vector'


def hash_config(config_dict: Dict[str, Any]) -> str:
    """Create hash of config for caching"""
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_cached_evaluation(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load cached evaluation results if they exist"""
    cache_path = config_path.parent / 'evaluation_cache.pkl'
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load cached evaluation: {e}")
    return None


def save_cached_evaluation(eval_data: Dict[str, Any], true_data: Dict[str, Any], metrics: Dict[str, Any], config_path: Path):
    """Save evaluation results and metrics to cache"""
    cache_data = {
        'eval_data': eval_data,
        'true_data': true_data,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save pickle cache
    cache_path = config_path.parent / 'evaluation_cache.pkl'
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    # Save metrics as YAML for easy reading
    metrics_path = config_path.parent / 'metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
        
    logging.info(f"Saved evaluation cache to {cache_path}")
    logging.info(f"Saved metrics to {metrics_path}")


def evaluate_model(model: torch.nn.Module, bridge: Any, dataset: Any, 
                  config_path: Optional[Path] = None, n_samples: int = 1000,
                  force_regenerate: bool = False, n_steps: int = 10) -> Dict[str, Any]:
    """
    Smart evaluation dispatcher with caching and data type detection
    
    Args:
        model: Trained model
        bridge: Bridge object  
        dataset: Dataset to evaluate on
        config_path: Path to config for caching (optional)
        n_samples: Number of samples to generate
        force_regenerate: Force regeneration even if cache exists
        n_steps: Optional override for number of sampling steps
        
    Returns:
        Dictionary containing evaluation data and metrics
    """
    # Check cache first (skip cache if using custom n_steps)
    if config_path and not force_regenerate:
        cached_results = load_cached_evaluation(config_path)
        if cached_results:
            logging.info("Using cached evaluation results")
            return cached_results
    
    # Detect data type
    data_type = detect_data_type(dataset)
    logging.info(f"Detected data type: {data_type}")
    
    # Generate evaluation data 
    eval_data = generate_evaluation_data(model, bridge, dataset, n_samples, n_steps)
    
    # Generate true distribution data by sampling bridge directly
    true_data = generate_true_trajectory_data(bridge, dataset, n_samples, n_steps)
    
    # Compute evaluation metrics
    metrics = compute_evaluation_metrics(eval_data)
    
    # Add sampling info if using custom steps
    metrics['n_sampling_steps'] = n_steps
    
    # Always save metrics to config directory if config_path provided
    if config_path:
        # Save metrics as YAML for easy reading
        metrics_path = config_path.parent / 'metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # Cache evaluation data only if using default settings
        save_cached_evaluation(eval_data, true_data, metrics, config_path)
    
    return {
        'eval_data': eval_data,
        'true_data': true_data,
        'metrics': metrics,
        'data_type': data_type
    }


def generate_evaluation_data(
    model: torch.nn.Module, 
    bridge: Any, 
    dataset: Any, 
    n_samples: int = 1000, 
    n_steps: int = 10
) -> Dict[str, np.ndarray]:
    """Generate evaluation data from trained model"""
    model.eval()
    
    # Get data efficiently using DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=False)
    batch = next(iter(dataloader))
    
    x0_target = batch['x_0'].squeeze(0).numpy()# .reshape(-1, batch['x_0'].shape[-1])
    x1_source = batch['x_1'].squeeze(0).numpy()# .reshape(-1, batch['x_1'].shape[-1])
    
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
        }
        if 'z' in batch:
            z = batch['z'].squeeze(0).numpy()# .reshape(-1, batch['z'].shape[-1])
            sampler_kwargs['z'] = {'z': torch.tensor(z).cuda()}
        
        # Add target mean for mean constrained bridges
        if is_mean_constrained:
            mu_0 = x0_torch.float().mean(dim=0)
            sampler_kwargs['mu_0'] = mu_0
            logging.info("Using mean constrained bridge with target mean")
        
        # Add n_steps to sampler
        sampler_kwargs['n_steps'] = n_steps
            
        # Generate samples with trajectories
        result = bridge.sampler(**sampler_kwargs)
        
        # Our bridge sampler always returns (x0_generated, x_trajectory, x_hat_trajectory)
        x0_generated, x_trajectory, x_hat_trajectory = result
        
        # Convert to numpy (handles both torch tensors and cupy arrays)
        def to_numpy(x):
            if x is None:
                return None
            if hasattr(x, 'get'):  # CuPy array
                return x.get()
            elif hasattr(x, 'cpu'):  # PyTorch tensor
                return x.cpu().numpy()
            else:
                return np.array(x)
        
        x0_generated = to_numpy(x0_generated)
        x_trajectory = to_numpy(x_trajectory)
        x_hat_trajectory = to_numpy(x_hat_trajectory)
    
    result_dict = {
        'x0_target': x0_target,
        'x1_batch': x1_source,
        'x0_generated': x0_generated,
    }
    
    # Only include trajectories if they exist
    if x_trajectory is not None:
        result_dict['x_trajectory'] = x_trajectory
    else:
        # Create minimal trajectory from start and end points
        result_dict['x_trajectory'] = np.array([x1_source, x0_generated])
        
    if x_hat_trajectory is not None:
        result_dict['x_hat_trajectory'] = x_hat_trajectory
    else:
        # Create minimal x_hat trajectory (just final prediction)
        result_dict['x_hat_trajectory'] = np.array([x0_generated])
        

    
    return result_dict


def compute_evaluation_metrics(eval_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics using new metrics module"""
    from metrics import compute_comprehensive_metrics
    
    # Get comprehensive metrics
    metrics = compute_comprehensive_metrics(eval_data)
    
    # Add basic compatibility metrics
    x0_generated = eval_data['x0_generated']
    x0_target = eval_data['x0_target']
    
    gen_mean = np.mean(x0_generated, axis=0)
    target_mean = np.mean(x0_target, axis=0)
    
    metrics['target_mean'] = target_mean.tolist()
    metrics['gen_mean'] = gen_mean.tolist()
    
    return metrics


def generate_true_trajectory_data(bridge: Any, dataset: Any, n_samples: int = 1000, n_steps: int = 10) -> Dict[str, np.ndarray]:
    """
    Generate true trajectory data by sampling bridge directly at each time step.
    This is completely bridge-agnostic and doesn't require any dummy models.
    """
    from torch.utils.data import DataLoader
    
    # Get data from dataset
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    batch = next(iter(dataloader))
    
    x0_true = batch['x_0'].squeeze(0).numpy()# .reshape(-1, batch['x_0'].shape[-1])
    x1_true = batch['x_1'].squeeze(0).numpy()# .reshape(-1, batch['x_1'].shape[-1])
    
    # Convert to torch tensors
    x0_torch = torch.tensor(x0_true).cuda() if torch.cuda.is_available() else torch.tensor(x0_true)
    x1_torch = torch.tensor(x1_true).cuda() if torch.cuda.is_available() else torch.tensor(x1_true)
    
    
    # Create time points from 1 to 0
    times = np.linspace(0.0, 1.0, n_steps + 1)
    
    # Sample bridge trajectory directly
    x_trajectory = []
    x_hat_trajectory = []
    
    with torch.no_grad():
        for i, t in enumerate(times):
            # Sample x_t from bridge at time t
            t, x_t, _ = bridge(x1_torch, x0_torch, t)
            x_trajectory.append(x_t.cpu().numpy())
            
            # For x_hat, the "true" prediction is just x0 (what perfect model should predict)
            x_hat_trajectory.append(x0_torch.cpu().numpy())
    
    # Convert to numpy arrays
    x_trajectory = np.array(x_trajectory)  # [T, B, d]
    x_hat_trajectory = np.array(x_hat_trajectory)  # [T, B, d]
    
    return {
        'x0_target': x0_true,
        'x1_batch': x1_true,
        'x0_generated': x0_true,  # True data is the target
        'x_trajectory': x_trajectory,
        'x_hat_trajectory': x_hat_trajectory,
    }

# Sampling sweep functionality moved to run_sampling_sweep.py

def log_evaluation_summary(eval_data: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
    """Log evaluation summary - dynamically iterate over available metrics"""
    logging.info("=== Evaluation Summary ===")
    
    # Log basic info first
    if 'data_type' in metrics:
        logging.info(f"Data type: {metrics['data_type']}")
    if 'n_samples' in metrics:
        logging.info(f"Samples: {metrics['n_samples']}")
    if 'n_dimensions' in metrics:
        logging.info(f"Dimensions: {metrics['n_dimensions']}")
    if 'image_shape' in metrics:
        logging.info(f"Image shape: {metrics['image_shape']}")
    
    # Log all other metrics dynamically
    skip_keys = {'data_type', 'n_samples', 'n_dimensions', 'image_shape', 'target_mean', 'gen_mean'}
    for metric_name, value in metrics.items():
        if metric_name not in skip_keys:
            if isinstance(value, float) and not np.isnan(value):
                logging.info(f"{metric_name}: {value:.4f}")
            elif isinstance(value, int):
                logging.info(f"{metric_name}: {value}")
    
    logging.info(f"Generated {len(eval_data['x0_generated'])} samples") 