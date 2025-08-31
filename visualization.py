"""
Clean and simple visualization module.
Beautiful plotting functions for any data type.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import scipy.stats

# Set style
plt.style.use('default')
sns.set_palette("husl")


def detect_plot_type(eval_data: Dict[str, Any]) -> str:
    """Detect if we should use vector, image, or specific dimensional plotting"""
    x0_shape = eval_data['x0_target'].shape
    
    if len(x0_shape) == 2:  # [samples, features]
        d = x0_shape[1]
        if d == 1:
            return '1d'
        elif d == 2:
            return '2d'
        else:
            return 'vector'
    elif len(x0_shape) == 4:  # [samples, channels, height, width]
        return 'image'
    else:
        return 'vector'  # Default fallback


def plot_loss_curve(losses: List[float], title: str = "Training Loss") -> plt.Figure:
    """Plot the training loss curve"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(losses, color='blue', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_1d_beautiful(samples: np.ndarray, title: str = "1D Distribution", true_samples: np.ndarray = None) -> plt.Figure:
    """Beautiful 1D plotting with optional true distribution comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot generated samples
    ax.hist(samples[:, 0], bins=50, alpha=0.7, density=True, color='blue', label='Generated')
    
    # Plot true samples if provided
    if true_samples is not None:
        ax.hist(true_samples[:, 0], bins=50, alpha=0.7, density=True, color='green', label='True')
        ax.legend()
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_2d_beautiful(samples: np.ndarray, title: str = "2D Distribution", true_samples: np.ndarray = None) -> plt.Figure:
    """Beautiful 2D plotting with scatter + marginals and optional true distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Main scatter plot
    ax_main = axes[1, 0]
    ax_main.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=15, color='blue', label='Generated')
    if true_samples is not None:
        ax_main.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, s=15, color='green', label='True')
        ax_main.legend()
    ax_main.set_xlabel('Dimension 0')
    ax_main.set_ylabel('Dimension 1')
    ax_main.grid(True, alpha=0.3)
    
    # Top marginal
    ax_top = axes[0, 0]
    ax_top.hist(samples[:, 0], bins=30, alpha=0.7, density=True, color='blue', label='Generated')
    if true_samples is not None:
        ax_top.hist(true_samples[:, 0], bins=30, alpha=0.7, density=True, color='green', label='True')
    ax_top.set_ylabel('Density')
    ax_top.set_title('Dimension 0')
    ax_top.grid(True, alpha=0.3)
    
    # Right marginal
    ax_right = axes[1, 1]
    ax_right.hist(samples[:, 1], bins=30, alpha=0.7, density=True, 
                 orientation='horizontal', color='blue', label='Generated')
    if true_samples is not None:
        ax_right.hist(true_samples[:, 1], bins=30, alpha=0.7, density=True, 
                     orientation='horizontal', color='green', label='True')
    ax_right.set_xlabel('Density')
    ax_right.set_title('Dimension 1')
    ax_right.grid(True, alpha=0.3)
    
    # Empty top-right
    axes[0, 1].axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_vector_beautiful(samples: np.ndarray, title: str = "High-D Distribution", true_samples: np.ndarray = None) -> plt.Figure:
    """Beautiful high-dimensional plotting with pairplot"""
    n_samples, d = samples.shape
    
    # Select dimensions to plot
    if d <= 5:
        dims_to_plot = list(range(d))
    else:
        dims_to_plot = [0, 1, d//2, d-2, d-1]  # First, middle, last
    
    # Create combined DataFrame
    data_list = []
    for i in range(min(n_samples, 800)):  # Limit for performance
        row = {'type': 'Generated'}
        for dim in dims_to_plot:
            row[f'dim_{dim}'] = samples[i, dim]
        data_list.append(row)
    
    # Add true samples if provided
    if true_samples is not None:
        n_true = min(len(true_samples), 800)
        for i in range(n_true):
            row = {'type': 'True'}
            for dim in dims_to_plot:
                row[f'dim_{dim}'] = true_samples[i, dim]
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # Create pairplot
    vars_to_plot = [f'dim_{dim}' for dim in dims_to_plot]
    if true_samples is not None:
        g = sns.pairplot(df, vars=vars_to_plot, hue='type', 
                        plot_kws={'s': 8, 'alpha': 0.7},
                        palette={'Generated': 'blue', 'True': 'green'})
    else:
        g = sns.pairplot(df, vars=vars_to_plot, plot_kws={'s': 10, 'alpha': 0.6, 'color': 'blue'})
    
    g.fig.suptitle(title, y=1.02, fontsize=16)
    return g.fig


def plot_image_beautiful(samples: np.ndarray, title: str = "Images") -> plt.Figure:
    """Beautiful image plotting"""
    return plot_image_samples(samples, title=title)


def plot_image_samples(
    images: np.ndarray, 
    title: str = "Image Samples",
    max_display: int = 64,
    cmap: str = 'gray'
) -> plt.Figure:
    """
    Display image samples in a grid.
    
    Args:
        images: Array of shape [N, H, W] or [N, C, H, W] or [N, H*W]
        title: Plot title
        max_display: Maximum number of images to display
        cmap: Colormap for grayscale images
    """
    # Handle different image formats
    if len(images.shape) == 2:  # [N, H*W] - reshape to square
        n_samples, flat_size = images.shape
        img_size = int(np.sqrt(flat_size))
        if img_size * img_size == flat_size:
            images = images.reshape(n_samples, img_size, img_size)
        else:
            raise ValueError(f"Cannot reshape flat images of size {flat_size} to square")
    
    n_samples = min(len(images), max_display)
    
    # Handle different image shapes
    if len(images.shape) == 4:  # [N, C, H, W]
        if images.shape[1] == 1:  # Grayscale
            images = images[:n_samples, 0]  # Remove channel dimension
        else:  # RGB
            images = images[:n_samples].transpose(0, 2, 3, 1)  # [N, H, W, C]
    else:  # [N, H, W]
        images = images[:n_samples]
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = [[axes]]
    elif grid_size > 1 and not isinstance(axes[0], np.ndarray):
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            ax = axes[i][j] if grid_size > 1 else axes[0][0]
            
            if idx < n_samples:
                img = images[idx]
                if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                    ax.imshow(img)
                else:  # Grayscale
                    ax.imshow(img, cmap=cmap)
                ax.set_title(f'Sample {idx}')
            
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_samples_at_time(
    samples: np.ndarray,
    title: str = "Samples",
    data_type: str = None,
    true_samples: np.ndarray = None
) -> plt.Figure:
    """
    Universal beautiful plotting function that auto-detects data type.
    
    Args:
        samples: Generated samples
        title: Plot title
        data_type: Override data type detection ('1d', '2d', 'vector', 'image')
        true_samples: True distribution samples for comparison
    """
    # Detect data type if not provided
    if data_type is None:
        if len(samples.shape) == 2:
            d = samples.shape[1]
            if d == 1:
                data_type = '1d'
            elif d == 2:
                data_type = '2d'
            else:
                data_type = 'vector'
        elif len(samples.shape) >= 3:
            data_type = 'image'
        else:
            raise ValueError(f"Unsupported data type: {samples.shape}")
    
    # Route to appropriate beautiful plotting function
    if data_type == '1d':
        return plot_1d_beautiful(samples, title, true_samples)
    elif data_type == '2d':
        return plot_2d_beautiful(samples, title, true_samples)
    elif data_type == 'image':
        return plot_image_beautiful(samples, title)
    else:  # vector
        return plot_vector_beautiful(samples, title, true_samples)


def plot_trajectory_evolution(
    eval_data: Dict[str, Any],
    trajectory_key: str = 'x_trajectory',
    title_prefix: str = "x_t Evolution",
    n_timesteps: int = 5,
    true_data: Dict[str, Any] = None
) -> Dict[str, plt.Figure]:
    """
    Plot trajectory evolution with true distribution comparison.
    
    Args:
        eval_data: Evaluation data dictionary
        trajectory_key: Which trajectory to plot ('x_trajectory' or 'x_hat_trajectory')
        title_prefix: Prefix for plot titles
        n_timesteps: Number of time points to show
        true_data: True distribution data for comparison
        
    Returns:
        Dictionary of figures for each time point
    """
    if trajectory_key not in eval_data:
        return {}
    
    trajectory = eval_data[trajectory_key]  # [T, B, ...]
    T = len(trajectory)
    
    # Select time indices
    if n_timesteps >= T:
        time_indices = list(range(T))
    else:
        time_indices = [int(i * (T-1) / (n_timesteps-1)) for i in range(n_timesteps)]
    
    figures = {}
    data_type = detect_plot_type(eval_data)
    
    for i, t_idx in enumerate(time_indices):
        time_val = 1.0 - (t_idx / (T-1))  # Convert to flow time (1→0)
        samples = trajectory[t_idx]  # [B, ...]
        
        # Get true samples at this time if available
        true_samples = None
        if true_data is not None and trajectory_key in true_data:
            true_traj = true_data[trajectory_key]
            if t_idx < len(true_traj):
                true_samples = true_traj[t_idx]
        
        title = f"{title_prefix} at t={time_val:.2f}"
        
        # For images, create separate true distribution file
        if data_type == 'image' and true_samples is not None:
            # Main figure with generated samples
            fig = plot_samples_at_time(samples, title=title, data_type=data_type)
            figures[f't_{time_val:.2f}'] = fig
            
            # Separate figure with true samples
            true_title = f"True Distribution at t={time_val:.2f}"
            true_fig = plot_samples_at_time(true_samples, title=true_title, data_type=data_type)
            figures[f't_{time_val:.2f}_true'] = true_fig
        else:
            # For vector data, include true samples on same plot
            fig = plot_samples_at_time(samples, title=title, data_type=data_type, true_samples=true_samples)
            figures[f't_{time_val:.2f}'] = fig
    
    return figures


def plot_x_t_trajectories(
    eval_data: Dict[str, Any], 
    title: str = "x_t Flow Trajectories",
    true_data: Dict[str, Any] = None
) -> Dict[str, plt.Figure]:
    """
    Plot x_t trajectory evolution using time-slice approach.
    Much more informative than line plots for high-dimensional data.
    """
    return plot_trajectory_evolution(
        eval_data=eval_data,
        trajectory_key='x_trajectory',
        title_prefix="x_t Flow",
        n_timesteps=5,
        true_data=true_data
    )


def plot_x_hat_trajectories(
    eval_data: Dict[str, Any], 
    title: str = "x̂₀ Model Predictions",
    true_data: Dict[str, Any] = None
) -> Dict[str, plt.Figure]:
    """
    Plot x̂₀ prediction evolution using time-slice approach.
    Shows model predictions at different time points.
    """
    return plot_trajectory_evolution(
        eval_data=eval_data,
        trajectory_key='x_hat_trajectory',
        title_prefix="x̂₀ Predictions",
        n_timesteps=5,
        true_data=true_data
    )


def plot_model_samples(
    eval_data: Dict[str, Any], 
    title: str = "Model Evaluation",
    true_data: Dict[str, Any] = None
) -> Dict[str, plt.Figure]:
    """
    Beautiful, simple plotting that auto-detects data type and includes true distributions.
    
    Args:
        eval_data: Dictionary from eval.generate_evaluation_data()
        title: Base title for plots
        true_data: True distribution data for comparison
        
    Returns:
        Dictionary of figures - simple and informative
    """
    figures = {}
    
    # Detect plot type
    plot_type = detect_plot_type(eval_data)
    logging.info(f"Using {plot_type} plotting for evaluation")
    
    # For any data type, we just need beautiful trajectory evolution
    x_t_figs = plot_x_t_trajectories(eval_data, f"{title} - x_t Trajectories", true_data)
    x_hat_figs = plot_x_hat_trajectories(eval_data, f"{title} - x̂₀ Predictions", true_data)
    
    # Flatten the trajectory figures into main dict
    for key, fig in x_t_figs.items():
        figures[f'x_t_{key}'] = fig
    for key, fig in x_hat_figs.items():
        figures[f'x_hat_{key}'] = fig
    
    # Add simple final distribution comparison
    if plot_type == 'image':
        figures['sample_grid'] = plot_image_beautiful(eval_data['x0_generated'], f"{title} - Generated")
        figures['target_grid'] = plot_image_beautiful(eval_data['x0_target'], f"{title} - Target")
    else:
        # For vector data, use beautiful plotting with true distribution if available
        true_samples = eval_data.get('x0_target', None)  # Use targets as "true" reference
        figures['final_distribution'] = plot_samples_at_time(
            eval_data['x0_generated'], 
            title=f"{title} - Final Distribution",
            data_type=plot_type,
            true_samples=true_samples
        )
    
    return figures


def plot_true_distributions(
    bridge: Any,
    dataset: Any,
    title: str = "True Distribution Analysis",
    n_samples: int = 1000
) -> Dict[str, plt.Figure]:
    """
    Generate and plot true distributions using dummy perfect model.
    """
    try:
        from eval import generate_true_distribution_data
        true_data = generate_true_distribution_data(bridge, dataset, n_samples=n_samples)
        
        # Use the same plotting system for true distributions
        return plot_model_samples(true_data, title=f"True - {title}")
        
    except Exception as e:
        logging.error(f"Could not generate true distribution plots: {e}")
        return {}


def save_plots(plots: Dict[str, plt.Figure], output_dir: str) -> None:
    """
    Save all plots to the specified directory.
    
    Args:
        plots: Dictionary of plot name -> matplotlib figure
        output_dir: Directory to save plots to
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for plot_name, fig in plots.items():
        # Create safe filename
        safe_name = plot_name.replace(" ", "_").replace("/", "_").replace(":", "")
        file_path = os.path.join(output_dir, f"{safe_name}.png")
        
        try:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Free memory
            logging.info(f"Saved plot: {file_path}")
        except Exception as e:
            logging.error(f"Failed to save plot {plot_name}: {e}")
    
    logging.info(f"All plots saved to: {output_dir}")