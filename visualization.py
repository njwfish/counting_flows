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
from matplotlib.colors import LinearSegmentedColormap

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


def plot_2d_trajectories(
    eval_data: Dict[str, Any], 
    title: str = "2D Flow Trajectories",
    n_trajectories: int = 50
) -> plt.Figure:
    """
    Clean 2D trajectory visualization showing actual paths from start to end.
    
    Args:
        eval_data: Dictionary from evaluation containing trajectories and targets
        title: Plot title
        n_trajectories: Number of individual trajectories to show
    """
    # Extract data
    x0_target = eval_data['x0_target']  # [B, 2]
    trajectory = eval_data['x_trajectory']  # [T, B, 2]
    
    if x0_target.shape[1] != 2:
        # Not 2D data, return empty figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"2D plot not applicable\n(data is {x0_target.shape[1]}D)", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Single plot for trajectories
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)
    
    # Select random trajectories to display
    n_show = min(n_trajectories, trajectory.shape[1])
    indices = np.random.choice(trajectory.shape[1], n_show, replace=False)
    
    # Create colormap for time progression (blue to red)
    cmap = plt.cm.plasma
    
    for i in indices:
        traj_path = trajectory[:, i, :]  # [T, 2]
        T = len(traj_path)
        
        # Plot complete trajectory as connected line with color progression
        colors = [cmap(t / (T - 1)) for t in range(T)]
        
        # Plot the full trajectory line
        ax.plot(traj_path[:, 0], traj_path[:, 1], 
               color=colors[T//2], alpha=0.4, linewidth=1, zorder=1)
        
        # Add color-coded points along the trajectory
        for t in range(0, T, max(1, T//10)):  # Sample points along trajectory
            ax.scatter(traj_path[t, 0], traj_path[t, 1], 
                      c=[colors[t]], s=8, alpha=0.6, zorder=2)
    
    # Add distinctive start and end markers for subset of trajectories
    n_markers = min(15, n_show)
    marker_indices = indices[:n_markers]
    
    start_points = trajectory[0, marker_indices, :]  # [n_markers, 2]
    end_points = trajectory[-1, marker_indices, :]   # [n_markers, 2]
    
    ax.scatter(start_points[:, 0], start_points[:, 1], 
              c='blue', s=80, marker='o', alpha=0.9, 
              label='Start (t=1)', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter(end_points[:, 0], end_points[:, 1], 
              c='red', s=80, marker='s', alpha=0.9, 
              label='End (t=0)', zorder=5, edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Dimension 0', fontsize=12)
    ax.set_ylabel('Dimension 1', fontsize=12) 
    ax.set_title('Sample Flow Trajectories', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_2d_time_evolution(
    eval_data: Dict[str, Any], 
    title: str = "2D Evolution Over Time"
) -> plt.Figure:
    """
    Show x_t and x̂ distributions at key time points in 2x4 grid.
    
    Args:
        eval_data: Dictionary from evaluation containing trajectories
        title: Plot title
    """
    # Extract data
    x0_target = eval_data['x0_target']  # [B, 2]
    x_trajectory = eval_data['x_trajectory']  # [T, B, 2]
    x_hat_trajectory = eval_data.get('x_hat_trajectory', None)  # [T, B, 2]
    
    if x0_target.shape[1] != 2:
        # Not 2D data, return empty figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"2D plot not applicable\n(data is {x0_target.shape[1]}D)", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Create 2x4 subplot layout
    has_xhat = x_hat_trajectory is not None
    n_rows = 2 if has_xhat else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 8 if has_xhat else 4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(title, fontsize=16)
    
    # Time points: t=1, 0.75, 0.25, 0
    time_values = [1.0, 0.75, 0.25, 0.0]
    time_indices = [
        0,  # t=1
        len(x_trajectory) // 4,  # t=0.75
        3 * len(x_trajectory) // 4,  # t=0.25  
        -1  # t=0
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Top row: x_t distributions
    for i, (t_val, t_idx, color) in enumerate(zip(time_values, time_indices, colors)):
        ax = axes[0, i]
        points = x_trajectory[t_idx, :, :]  # [B, 2]
        
        ax.scatter(points[:, 0], points[:, 1], 
                  c=color, alpha=0.3, s=10)
        
        ax.set_xlabel('Dimension 0', fontsize=10)
        if i == 0:
            ax.set_ylabel('Dimension 1', fontsize=10)
        ax.set_title(f'x_t at t={t_val}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Bottom row: x̂ distributions (if available)
    if has_xhat:
        for i, (t_val, t_idx, color) in enumerate(zip(time_values, time_indices, colors)):
            ax = axes[1, i]
            points = x_hat_trajectory[t_idx, :, :]  # [B, 2]
            
            ax.scatter(points[:, 0], points[:, 1], 
                      c=color, alpha=0.3, s=10)
            
            ax.set_xlabel('Dimension 0', fontsize=10)
            if i == 0:
                ax.set_ylabel('Dimension 1', fontsize=10)
            ax.set_title(f'x̂ at t={t_val}', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_full_reverse_trajectories(
    eval_data: Dict[str, Any],
    title: str = "Full Reverse Trajectories"
) -> plt.Figure:
    """
    Plot reverse trajectories for all samples with elegant dict-based plotting.
    
    Args:
        eval_data: Dict from evaluation containing trajectories and targets
        title: Plot title
    """
    # Extract basic data
    x0_target = eval_data['x0_target'] 
    x1_batch = eval_data['x1_batch']
    trajectory = eval_data['x_trajectory']
    x_hat_trajectory = eval_data['x_hat_trajectory']
    
    # Check what trajectory data we have
    trajectory_types = ['x_trajectory', 'x_hat_trajectory']
    if 'M_trajectory' in eval_data:
        trajectory_types.append('M_trajectory')
    
    # Create subplot grid based on available data
    n_rows = len(trajectory_types)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array
    fig.suptitle(title, fontsize=16)
    
    B, d = x0_target.shape
    # Limit dimensions intelligently - show first few dimensions for high-d data
    if d <= 4:
        n_dims_to_plot = d
        dim_indices = list(range(d))
    else:
        n_dims_to_plot = 4  
        # For high-d data, show first 2 and last 2 dimensions
        dim_indices = [0, 1, d-2, d-1]
    
    # Plot configurations for each trajectory type
    plot_configs = {
        'x_trajectory': {'color': 'blue', 'label': 'x_t', 'title': 'x_t Trajectories'},
        'x_hat_trajectory': {'color': 'purple', 'label': 'x̂₀', 'title': 'x̂₀ Predictions'},
        'M_trajectory': {'color': 'orange', 'label': 'M_t', 'title': 'M_t Trajectories'}
    }
    
    # Plot each trajectory type
    for row_idx, traj_key in enumerate(trajectory_types):
        traj_data = eval_data[traj_key]
        time_steps = np.linspace(1.0, 0.0, len(traj_data))
        config = plot_configs[traj_key]
        
        for plot_dim_idx in range(n_dims_to_plot):
            dim = dim_indices[plot_dim_idx]  # Actual dimension index
            ax = axes[row_idx, plot_dim_idx]
            
            # Plot individual trajectories with low alpha
            for b in range(min(B, 1000)):  # Limit for performance
                alpha = 0.1 if B > 100 else 0.5
                ax.plot(time_steps, traj_data[:, b, dim], config['color'], alpha=alpha, linewidth=0.5)
            
            # Plot percentiles for summary statistics
            percentiles = np.percentile(traj_data[:, :, dim], [10, 25, 50, 75, 90], axis=1)
            mean = np.mean(traj_data[:, :, dim], axis=1)
            ax.plot(time_steps, mean, config['color'], linewidth=2, label=f"Mean {config['label']}")
            ax.fill_between(time_steps, percentiles[1], percentiles[3], 
                           alpha=0.4, color=config['color'], label='25-75%')
            ax.fill_between(time_steps, percentiles[0], percentiles[4], 
                           alpha=0.3, color=config['color'], label='10-90%')
            
            # Add reference lines for x_t and x̂₀ trajectories
            if traj_key in ['x_trajectory', 'x_hat_trajectory']:
                target_mean = x0_target[:, dim].mean()
                x1_mean = x1_batch[:, dim].mean()
                ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
                if traj_key == 'x_trajectory':
                    ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
            
            ax.set_title(f"{config['title']} - Dimension {dim}")
            ax.set_xlabel('Time t (1→0)')
            ax.set_ylabel(f"{config['label']} value")
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
    
    # Collect all bridge inputs over time
    bridge_inputs_over_time = {}
    
    for t_val in times:
        # Sample from bridge at this time point
        x0_sample = torch.tensor(x0_batch).cuda()
        x1_sample = torch.tensor(x1_batch).cuda()
        
        # Sample from bridge (returns standardized dict format)
        bridge_output = bridge(x0_sample, x1_sample, t_target=t_val)
        inputs = bridge_output['inputs']
        
        # Store each input type
        for key, value in inputs.items():
            if key not in bridge_inputs_over_time:
                bridge_inputs_over_time[key] = []
            # Convert to numpy
            bridge_inputs_over_time[key].append(value.cpu().numpy())
    
    # Convert to arrays for plotting
    input_arrays = {}
    for key, values in bridge_inputs_over_time.items():
        input_arrays[key] = np.array(values)  # [T, B, d]
    
    # Plot configurations
    plot_configs = {
        'x_t': {'color': 'blue', 'label': 'x_t', 'title': 'x_t Evolution'},
        'M_t': {'color': 'orange', 'label': 'M_t', 'title': 'M_t Evolution'}
    }
    
    # Determine what to plot (skip 't' as it's just time, skip 'z' if present)
    plot_keys = [key for key in input_arrays.keys() if key not in ['t', 'z']]
    
    # Add reference interpolation if we have x_t
    if 'x_t' in input_arrays:
        plot_keys.insert(1, 'reference')  # Add after x_t
    
    # Create subplot layout
    n_rows = len(plot_keys)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(title, fontsize=16)
    
    # Plot each input type
    for row_idx, plot_key in enumerate(plot_keys):
        if plot_key == 'reference':
            # Special case: linear interpolation reference
            for dim in range(n_dims_to_plot):
                ax = axes[row_idx, dim]
                
                target_mean = x0_batch[:, dim].mean()
                x1_mean = x1_batch[:, dim].mean()
                linear_interp = (1 - times) * target_mean + times * x1_mean
                
                ax.plot(times, linear_interp, 'purple', linewidth=2, label='Linear interpolation')
                ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
                
                ax.set_title(f'Reference Interpolation - Dimension {dim}')
                ax.set_xlabel('Time t')
                ax.set_ylabel('Expected value')
                ax.grid(True, alpha=0.3)
                if dim == 0:
                    ax.legend(fontsize='small', loc='best')
        else:
            # Regular input plotting
            data = input_arrays[plot_key]
            config = plot_configs.get(plot_key, {'color': 'black', 'label': plot_key, 'title': f'{plot_key} Evolution'})
            
            for dim in range(n_dims_to_plot):
                ax = axes[row_idx, dim]
                
                # Plot individual trajectories 
                for i in range(min(data.shape[1], 100)):  # Limit for performance
                    ax.plot(times, data[:, i, dim], config['color'], alpha=0.3, linewidth=1)
                
                # Plot mean trajectory
                mean_traj = np.mean(data[:, :, dim], axis=1)
                ax.plot(times, mean_traj, config['color'], linewidth=3, label=f"Mean {config['label']}")
                
                # Add reference lines for x_t
                if plot_key == 'x_t':
                    target_mean = x0_batch[:, dim].mean()
                    x1_mean = x1_batch[:, dim].mean()
                    ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
                    ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
                
                ax.set_title(f"{config['title']} - Dimension {dim}")
                ax.set_xlabel('Time t')
                ax.set_ylabel(f"{config['label']} value")
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
        Dictionary with 'trajectories', 'distributions', and optionally '2d_trajectories' and '2d_time_evolution' figures
    """
    figures = {}
    
    # Check if this is 2D data
    data_dim = eval_data['x0_target'].shape[1]
    is_2d = (data_dim == 2)
    
    # 1. Trajectory plot (always included)
    figures['trajectories'] = plot_full_reverse_trajectories(
        eval_data=eval_data,
        title=f"{title} - Trajectories"
    )
    
    # 2. Distribution comparison (always included)
    figures['distributions'] = plot_distribution_comparison(
        x0_target=eval_data['x0_target'],  # Target is x0 (what we want to generate)
        x1_source=eval_data['x1_batch'],  # Source is x1 (what we start reverse sampling from)
        x0_generated=eval_data['x0_generated'],  # Generated is our model output
        title=f"{title} - Distribution Comparison"
    )
    
    # 3. Special 2D plots (only for 2D data)
    if is_2d:
        figures['2d_trajectories'] = plot_2d_trajectories(
            eval_data=eval_data,
            title=f"{title} - 2D Flow Trajectories"
        )
        figures['2d_time_evolution'] = plot_2d_time_evolution(
            eval_data=eval_data,
            title=f"{title} - 2D Time Evolution"
        )
        logging.info("Generated 2D trajectory and time evolution plots for 2D dataset")
    
    return figures


def save_plots(figs: Dict[str, plt.Figure], output_dir: str = "plots") -> None:
    """Save a dictionary of figures to files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, fig in figs.items():
        filepath = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot: {filepath}") 