"""
Sampling, evaluation, and plotting script for counting flows.
Clean Hydra-based CLI for generating samples with specified n_steps.
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import numpy as np
import pickle
import cupy as cp
from torch.utils.data import random_split
from typing import Dict, Any, Optional

# Capture original working directory before Hydra changes it
ORIGINAL_CWD = Path.cwd().resolve()

from visualization import plot_loss_curve, plot_model_samples, save_plots
from evaluate import evaluate_model

def run_sampling_evaluation(
    trained_model: torch.nn.Module,
    bridge: Any,
    dataset: Any,
    output_dir: Path,
    n_steps: int = 10,
    n_samples: int = 10000,
    n_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run sampling evaluation with specified parameters.
    
    Args:
        trained_model: Trained model
        bridge: Bridge object
        dataset: Evaluation dataset
        output_dir: Directory to save results
        n_steps: Number of sampling steps
        n_samples: Number of samples to generate
        n_epochs: Number of epochs to load
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Running sampling evaluation with {n_steps} steps...")
    
    # Create results directory for this n_steps
    results_dir = output_dir / f"n_steps={n_steps},n_epochs={n_epochs}"
    results_dir.mkdir(exist_ok=True)
    
    # Run evaluation
    eval_result = evaluate_model(
        trained_model, 
        bridge, 
        dataset,
        config_path=results_dir / "config.yaml",
        n_samples=n_samples,
        n_steps=n_steps
    )
    
    eval_data = eval_result['eval_data']
    true_data = eval_result['true_data']
    metrics = eval_result['metrics']
    
    # Log results
    logging.info("=== Sampling Evaluation Summary ===")
    logging.info(f"Steps: {n_steps}")
    logging.info(f"Samples: {n_samples}")
    logging.info(f"Epochs: {n_epochs}")
    for metric_name, value in metrics.items():
        if metric_name not in {'data_type', 'n_samples', 'n_dimensions', 'image_shape', 'target_mean', 'gen_mean'}:
            if isinstance(value, float) and not np.isnan(value):
                logging.info(f"{metric_name}: {value:.4f}")
    
    # Save evaluation data
    eval_data_path = results_dir / "eval_data.pkl"
    with open(eval_data_path, 'wb') as f:
        pickle.dump(eval_data, f)
    
    # Generate and save plots
    plots_dir = results_dir / "plots"
    if plots_dir.exists() and any(plots_dir.glob("*.png")):
        logging.info(f"Plots already exist at {plots_dir}, skipping plot generation")
    else:
        plots = {}
        
        # Generate model evaluation plots with true distributions
        sample_figs = plot_model_samples(eval_data, title=f"Samples (n_steps={n_steps})", true_data=true_data)
        plots.update(sample_figs)
        
        # Save plots
        save_plots(plots, str(plots_dir))
        logging.info(f"Plots saved to: {plots_dir}")
    
    return eval_result
