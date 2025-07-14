"""
Visualization Functions for Count-based Flow Matching

Provides comprehensive plotting and debugging visualization for count flows.
Updated to work with the new Hydra-based system and CuPy bridges.
"""

import torch
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
import seaborn as sns
import pandas as pd

def plot_loss_curve(losses: List[float], title: str = "Training Loss") -> plt.Figure:
    """Plot the training loss curve"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(losses)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add smoothed line
    if len(losses) > 100:
        window = len(losses) // 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window//2, len(losses) - window//2 + 1), smoothed, 'r-', linewidth=2, label='Smoothed')
        ax.legend()
    
    return fig


def plot_generation_analysis(
    x0_samples: torch.Tensor, x0_target: torch.Tensor, x1_batch: torch.Tensor, 
    title: str = "Generation Analysis"
) -> plt.Figure:
    """Analyze the quality of generated samples"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    d = min(4, x0_samples.shape[1])
    
    # Plot histograms for each dimension
    for dim in range(min(2, d)):
        ax = axes[0, dim]
        
        samples = x0_samples[:, dim].cpu().numpy()
        targets = x0_target[:, dim].cpu().numpy()
        
        ax.hist(samples, bins=30, alpha=0.7, density=True, label='Generated', color='blue')
        ax.hist(targets, bins=30, alpha=0.5, density=True, label='Target', color='red')
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Count value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Sample statistics comparison
    ax = axes[0, 2]
    sample_means = x0_samples.float().mean(0).cpu().numpy()
    target_means = x0_target.float().mean(0).cpu().numpy()
    
    dims = np.arange(len(sample_means))
    width = 0.35
    ax.bar(dims - width/2, sample_means, width, label='Generated mean', alpha=0.7)
    ax.bar(dims + width/2, target_means, width, label='Target mean', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean count')
    ax.set_title('Mean Comparison')
    ax.legend()
    
    # Variance comparison
    ax = axes[1, 0]
    sample_vars = x0_samples.float().var(0).cpu().numpy()
    target_vars = x0_target.float().var(0).cpu().numpy()
    
    ax.bar(dims - width/2, sample_vars, width, label='Generated var', alpha=0.7)
    ax.bar(dims + width/2, target_vars, width, label='Target var', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Comparison')
    ax.legend()
    
    # Scatter plot: generated vs target means
    ax = axes[1, 1]
    ax.scatter(target_means, sample_means, alpha=0.7)
    min_val, max_val = min(target_means.min(), sample_means.min()), max(target_means.max(), sample_means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect match')
    ax.set_xlabel('Target mean')
    ax.set_ylabel('Generated mean')
    ax.set_title('Mean Correlation')
    ax.legend()
    
    # Sample trajectory (first few samples)
    ax = axes[1, 2]
    n_show = min(50, len(x0_samples))
    for dim in range(min(2, d)):
        ax.plot(x0_samples[:n_show, dim].cpu().numpy(), alpha=0.7, label=f'Dim {dim}')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Count value')
    ax.set_title('Sample Trajectories')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_full_reverse_trajectories(
    trajectory: np.ndarray, 
    x_hat_trajectory: np.ndarray,
    M_trajectory: np.ndarray, 
    x0_target: np.ndarray, 
    x1_batch: np.ndarray, 
    title: str = "Full Reverse Trajectories"
) -> plt.Figure:
    """
    Plot reverse trajectories for all samples with elegant handling of large datasets.
    
    Args:
        trajectory: List of tensors, each of shape [B, d] representing x_t at each time step
        x_hat_trajectory: List of tensors, each of shape [B, d] representing x̂₀ predictions
        M_trajectory: List of tensors, each of shape [B, d] representing M_t at each time step
        x0_target: Tensor of shape [B, d] - target x0 values
        x1_batch: Tensor of shape [B, d] - starting x1 values  
        title: Plot title
    """
    # Always create 3x4 subplot grid for x_t, x̂₀, and M_t
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    B, d = x0_target.shape
    time_steps = np.linspace(1.0, 0.0, len(trajectory))  # Reverse time from 1 to 0
    x_hat_time_steps = np.linspace(1.0, 0.0, len(x_hat_trajectory))  # Match x_hat_array length
    M_time_steps = np.linspace(1.0, 0.0, len(M_trajectory))  # Match M_array length
    
    # Plot first 4 dimensions
    n_dims_to_plot = min(4, d)
    
    # Top row: x_t trajectories
    for dim in range(n_dims_to_plot):
        ax = axes[0, dim]
        
        # Plot all trajectories with low alpha for density visualization
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.1 if B > 100 else 0.5
            ax.plot(time_steps, trajectory[:, b, dim], 'b-', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for summary statistics
        traj_percentiles = np.percentile(trajectory[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(time_steps, traj_percentiles[2], 'r-', linewidth=2, label='Median x_t')
        ax.fill_between(time_steps, traj_percentiles[1], traj_percentiles[3], 
                       alpha=0.4, color='red', label='25-75%') 
        ax.fill_between(time_steps, traj_percentiles[0], traj_percentiles[4], 
                       alpha=0.3, color='red', label='10-90%')
        
        # Plot target mean as reference
        target_mean = x0_target[:, dim].mean()
        x1_mean = x1_batch[:, dim].mean()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
        
        ax.set_title(f'x_t Trajectories - Dimension {dim}')
        ax.set_xlabel('Time t (1→0)')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Show time going from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Second row: x̂₀ predictions
    for dim in range(n_dims_to_plot):
        ax = axes[1, dim]
        
        # Plot all x̂₀ predictions with low alpha
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.1 if B > 100 else 0.5
            ax.plot(x_hat_time_steps, x_hat_trajectory[:, b, dim], 'purple', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for x̂₀ predictions
        xhat_percentiles = np.percentile(x_hat_trajectory[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(x_hat_time_steps, xhat_percentiles[2], 'purple', linewidth=2, label='Median x̂₀')
        ax.fill_between(x_hat_time_steps, xhat_percentiles[1], xhat_percentiles[3], 
                       alpha=0.4, color='purple', label='25-75%')
        ax.fill_between(x_hat_time_steps, xhat_percentiles[0], xhat_percentiles[4], 
                       alpha=0.3, color='purple', label='10-90%')
        
        # Plot target mean as reference
        target_mean = x0_target[:, dim].mean()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        
        ax.set_title(f'x̂₀ Predictions - Dimension {dim}')
        ax.set_xlabel('Time t (1→0)')
        ax.set_ylabel('Predicted x₀')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Show time going from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Third row: M_t trajectories
    for dim in range(n_dims_to_plot):
        ax = axes[2, dim]
        
        # Plot all M_t trajectories with low alpha
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.5 if B > 100 else 0.5
            ax.plot(M_time_steps, M_trajectory[:, b, dim], 'orange', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for M_t
        M_percentiles = np.percentile(M_trajectory[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(M_time_steps, M_percentiles[2], 'orange', linewidth=2, label='Median M_t')
        ax.fill_between(M_time_steps, M_percentiles[1], M_percentiles[3], 
                       alpha=0.4, color='orange', label='25-75%')
        ax.fill_between(M_time_steps, M_percentiles[0], M_percentiles[4], 
                       alpha=0.3, color='orange', label='10-90%')
        
        ax.set_title(f'M_t Trajectories - Dimension {dim}')
        ax.set_xlabel('Time t (1→0)')
        ax.set_ylabel('M_t value')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Show time going from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    plt.tight_layout()
    return fig


def plot_bridge_marginals(x0_batch: torch.Tensor, x1_batch: torch.Tensor, 
                         bridge: Any, times: Optional[List[float]] = None, 
                         title: str = "Bridge Marginals") -> plt.Figure:
    """
    Plot true x_t and M_t marginal distributions from the bridge.
    
    Args:
        x0_batch: Tensor of shape [B, d] - x0 samples
        x1_batch: Tensor of shape [B, d] - x1 samples  
        bridge: CuPy bridge object
        times: List of time points to sample at (if None, uses dense sampling)
        n_samples: Number of samples to generate for each time point
        title: Plot title
    """
    # Use dense time sampling for smooth line plots
    if times is None:
        times = np.linspace(0.0, 1.0, 21)  # Dense sampling: 21 points from 0 to 1
    else:
        times = np.array(times)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    d = x0_batch.shape[1]
    n_dims_to_plot = min(4, d)
    
    # Collect samples for all time points
    all_x_t_samples = []  # List of [n_samples, d] arrays
    all_M_samples = []    # List of [n_samples, d] arrays
    
    for t_val in times:
        # Sample from bridge at this time point
        x0_sample = x0_batch.cuda()
        x1_sample = x1_batch.cuda()
        
        # Sample from bridge 
        x_t, M_t, _ = bridge(x0_sample, x1_sample, t_target=t_val)
        
        # Convert CuPy arrays to numpy
        x_t = x_t.cpu().numpy()
        M_t = M_t.cpu().numpy()
        
        all_x_t_samples.append(x_t)
        all_M_samples.append(M_t)
    
    # Convert to arrays for plotting: [time_steps, n_samples, d]
    x_t_array = np.array(all_x_t_samples)  # [T, B, d]
    M_array = np.array(all_M_samples)      # [T, B, d]
    
    # Top row: x_t evolution over time
    for dim in range(n_dims_to_plot):
        ax = axes[0, dim]
        
        # Plot individual trajectories with low alpha
        n_traj_show = x_t_array.shape[1]
        for i in range(n_traj_show):
            ax.plot(times, x_t_array[:, i, dim], 'b-', alpha=0.3, linewidth=1)
        
        # Plot median trajectory
        traj_median = np.median(x_t_array[:, :, dim], axis=1)
        ax.plot(times, traj_median, 'r-', linewidth=3, label='Median x_t')
        
        # Plot target mean as reference
        target_mean = x0_batch[:, dim].cpu().numpy().mean()
        x1_mean = x1_batch[:, dim].cpu().numpy().mean()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
        
        ax.set_title(f'x_t Evolution - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Middle row: Linear interpolation reference
    for dim in range(n_dims_to_plot):
        ax = axes[1, dim]
        
        # Plot linear interpolation between x0 and x1 means
        target_mean = x0_batch[:, dim].cpu().numpy().mean()
        x1_mean = x1_batch[:, dim].cpu().numpy().mean()
        linear_interp = (1 - times) * target_mean + times * x1_mean
        
        ax.plot(times, linear_interp, 'purple', linewidth=2, label='Linear interpolation')
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        
        ax.set_title(f'Reference Interpolation - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Expected value')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Bottom row: M_t evolution over time
    for dim in range(n_dims_to_plot):
        ax = axes[2, dim]
        
        # Plot percentiles over time
        M_percentiles = np.percentile(M_array[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(times, M_percentiles[2], 'orange', linewidth=2, label='Median M_t')
        ax.fill_between(times, M_percentiles[1], M_percentiles[3], 
                       alpha=0.4, color='orange', label='25-75%')
        ax.fill_between(times, M_percentiles[0], M_percentiles[4], 
                       alpha=0.3, color='orange', label='10-90%')
        
        ax.set_title(f'M_t Evolution - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('M_t value')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    plt.tight_layout()
    return fig



def plot_distribution_comparison(
    x0_target: np.ndarray, x1_source: np.ndarray, x0_generated: np.ndarray, 
    title: str = "Distribution Comparison"
) -> plt.Figure:
    """
    Create pairplot-style comparison of source, target, and generated distributions.
    
    Args:
        x0_target: Target samples (what we want to generate) [n_samples, d]
        x1_source: Source samples (what we start reverse sampling from) [n_samples, d]
        x0_generated: Generated samples (what our model produces) [n_samples, d]
        title: Plot title
    """
    # Data is already numpy
    x0_target_np = x0_target
    x1_source_np = x1_source
    x0_generated_np = x0_generated
    
    n_samples, d = x0_target_np.shape
    
    # Select dimensions to plot
    if d == 1:
        dims_to_plot = [0]
    elif d == 2:
        dims_to_plot = [0, 1]
    elif d <= 5:
        dims_to_plot = list(range(d))
    else:
        # Sample 5 random dimensions for high-dimensional data
        np.random.seed(42)  # For reproducibility
        dims_to_plot = sorted(np.random.choice(d, size=5, replace=False))
    
    n_dims_plot = len(dims_to_plot)
    
    # Special case for 2D data: create a nice scatter + marginal plot
    if n_dims_plot == 2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        
        dim1, dim2 = dims_to_plot
        
        # Main scatter plot
        ax_main = axes[1, 0]
        ax_main.scatter(x0_target_np[:, dim1], x0_target_np[:, dim2], 
                       alpha=0.6, s=20, label='Target', color='blue')
        ax_main.scatter(x1_source_np[:, dim1], x1_source_np[:, dim2], 
                       alpha=0.6, s=20, label='Source', color='red')
        ax_main.scatter(x0_generated_np[:, dim1], x0_generated_np[:, dim2], 
                       alpha=0.6, s=20, label='Generated', color='green')
        ax_main.set_xlabel(f'Dimension {dim1}')
        ax_main.set_ylabel(f'Dimension {dim2}')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Top marginal (dim1)
        ax_top = axes[0, 0]
        ax_top.hist(x0_target_np[:, dim1], bins=30, alpha=0.6, density=True, 
                   label='Target', color='blue')
        ax_top.hist(x1_source_np[:, dim1], bins=30, alpha=0.6, density=True, 
                   label='Source', color='red')
        ax_top.hist(x0_generated_np[:, dim1], bins=30, alpha=0.6, density=True, 
                   label='Generated', color='green')
        ax_top.set_ylabel('Density')
        ax_top.set_title(f'Dimension {dim1} Marginal')
        ax_top.legend()
        ax_top.grid(True, alpha=0.3)
        
        # Right marginal (dim2)
        ax_right = axes[1, 1]
        ax_right.hist(x0_target_np[:, dim2], bins=30, alpha=0.6, density=True, 
                     orientation='horizontal', label='Target', color='blue')
        ax_right.hist(x1_source_np[:, dim2], bins=30, alpha=0.6, density=True, 
                     orientation='horizontal', label='Source', color='red')
        ax_right.hist(x0_generated_np[:, dim2], bins=30, alpha=0.6, density=True, 
                     orientation='horizontal', label='Generated', color='green')
        ax_right.set_xlabel('Density')
        ax_right.set_title(f'Dimension {dim2} Marginal')
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)
        
        # Empty the top-right subplot
        axes[0, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    # For higher dimensions, create a pairplot using seaborn
    else:
        # Create combined dataset
        data_list = []
        
        # Add source data
        for i in range(n_samples):
            row = {'type': 'Target'}
            for j, dim in enumerate(dims_to_plot):
                row[f'dim_{dim}'] = x0_target_np[i, dim]
            data_list.append(row)
        
        # Add target data  
        for i in range(n_samples):
            row = {'type': 'Source'}
            for j, dim in enumerate(dims_to_plot):
                row[f'dim_{dim}'] = x1_source_np[i, dim]
            data_list.append(row)
        
        # Add generated data
        for i in range(n_samples):
            row = {'type': 'Generated'}
            for j, dim in enumerate(dims_to_plot):
                row[f'dim_{dim}'] = x0_generated_np[i, dim]
            data_list.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Create pairplot
        fig = plt.figure(figsize=(3*n_dims_plot, 3*n_dims_plot))
        
        # Use seaborn pairplot
        vars_to_plot = [f'dim_{dim}' for dim in dims_to_plot]
        g = sns.pairplot(df, vars=vars_to_plot, hue='type', 
                        palette={'Source': 'blue', 'Target': 'red', 'Generated': 'green'},
                        plot_kws={'s': 20, 'alpha': 0.6})
        
        g.fig.suptitle(title, y=1.02, fontsize=16)
        
        return g.fig

def plot_model_samples(
    eval_data: Dict[str, Any], 
    title: str = "Model Evaluation"
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive model evaluation plots from evaluation data.
    
    Args:
        eval_data: Dictionary from eval.generate_evaluation_data()
        title: Base title for plots
        
    Returns:
        Dictionary with 'trajectories' and 'distributions' figures
    """
    figures = {}
    
    # 1. Trajectory plot
    figures['trajectories'] = plot_full_reverse_trajectories(
        trajectory=eval_data['x_trajectory'],
        x_hat_trajectory=eval_data['x_hat_trajectory'],
        M_trajectory=eval_data['M_trajectory'],
        x0_target=eval_data['x0_target'],
        x1_batch=eval_data['x1_batch'],
        title=f"{title} - Trajectories"
    )
    
    # 2. Distribution comparison
    figures['distributions'] = plot_distribution_comparison(
        x0_target=eval_data['x0_target'],  # Target is x0 (what we want to generate)
        x1_source=eval_data['x1_batch'],  # Source is x1 (what we start reverse sampling from)
        x0_generated=eval_data['x0_generated'],  # Generated is our model output
        title=f"{title} - Distribution Comparison"
    )
    
    return figures


def save_plots(figs: Dict[str, plt.Figure], output_dir: str = "plots") -> None:
    """Save a dictionary of figures to files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figs.items():
        filepath = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot: {filepath}") 