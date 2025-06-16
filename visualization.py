"""
Visualization Functions for Count-based Flow Matching

Provides comprehensive plotting and debugging visualization for count flows.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .scheduling import make_time_spacing_schedule
from .samplers import bd_reverse_sampler, reflected_bd_reverse_sampler


def plot_schedule_comparison(K=50, title="Schedule Comparison"):
    """Visualize different r(t) and Φ(t) schedule types"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(title)
    
    # R schedules (for NB bridge)
    schedule_types = ["linear", "cosine", "exponential", "polynomial", "sqrt", "sigmoid"]
    
    for i, schedule_type in enumerate(schedule_types):
        if i >= 6:  # Only plot first 6 schedules
            break
            
        ax = axes[i//3, i%3]
        
        try:
            # Default parameters for each schedule
            kwargs = {}
            if schedule_type == "exponential":
                kwargs = {'growth_rate': 2.0}
            elif schedule_type == "polynomial":
                kwargs = {'power': 2.0}
            elif schedule_type == "sigmoid":
                kwargs = {'steepness': 10.0, 'midpoint': 0.5}
            
            phi_sched, R = make_phi_schedule(K, phi_min=0.0, phi_max=20.0, 
                                           schedule_type=schedule_type, **kwargs)
            times = torch.linspace(0, 1, K+1)
            
            # Plot Φ(t) schedule
            ax.plot(times.numpy(), phi_sched.numpy(), linewidth=2, label=f'Φ(t) {schedule_type}', color='blue')
            
            ax.set_title(f'{schedule_type.title()} Schedule')
            ax.set_xlabel('Time t')
            ax.set_ylabel('Φ(t) value', color='blue')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='blue')
            ax.legend(loc='upper left')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            ax.set_title(f'{schedule_type.title()} (Error)')
    
    # Add comparison plot showing different phi schedules
    if len(schedule_types) >= 2:
        ax = axes[2, 0]
        
        # Compare different phi schedules
        phi_linear, _ = make_phi_schedule(K, phi_min=0.0, phi_max=20.0, schedule_type="linear")
        phi_cosine, _ = make_phi_schedule(K, phi_min=0.0, phi_max=20.0, schedule_type="cosine")
        times = torch.linspace(0, 1, K+1)
        
        ax.plot(times.numpy(), phi_linear.numpy(), linewidth=2, label='Linear Φ(t)', color='blue')
        ax.plot(times.numpy(), phi_cosine.numpy(), linewidth=2, label='Cosine Φ(t)', color='green')
        ax.set_title('Comparison: Different Φ(t) Schedules')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Φ(t) value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_time_spacing_comparison(K=50, title="Time Spacing Comparison"):
    """Visualize different time spacing schedules"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    
    spacing_types = ["uniform", "early_dense", "late_dense", "middle_dense"]
    
    for i, spacing_type in enumerate(spacing_types):
        ax = axes[i//2, i%2]
        
        try:
            if spacing_type == "uniform":
                times = make_time_spacing_schedule(K, spacing_type)
            else:
                times = make_time_spacing_schedule(K, spacing_type, concentration=2.0)
            
            # Plot time points
            y_vals = torch.arange(len(times))
            ax.plot(times.numpy(), y_vals.numpy(), 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{spacing_type.replace("_", " ").title()}')
            ax.set_xlabel('Time value')
            ax.set_ylabel('Step index')
            ax.grid(True, alpha=0.3)
            
            # Also show step sizes
            if len(times) > 1:
                step_sizes = torch.diff(times)
                ax_twin = ax.twinx()
                ax_twin.plot(times[1:].numpy(), step_sizes.numpy(), 'r--', alpha=0.7, label='Step size')
                ax_twin.set_ylabel('Step size', color='red')
                ax_twin.tick_params(axis='y', labelcolor='red')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            ax.set_title(f'{spacing_type.title()} (Error)')
    
    plt.tight_layout()
    return fig



def plot_generation_analysis(x0_samples, x0_target, x1_batch, title="Generation Analysis"):
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


def plot_loss_curve(losses, title="Training Loss"):
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


def plot_full_reverse_trajectories(trajectory, x_hat_trajectory, M_trajectory, x0_target, x1_batch, mode, K, title="Full Reverse Trajectories"):
    """
    Plot reverse trajectories for all samples with elegant handling of large datasets.
    
    Args:
        trajectory: List of tensors, each of shape [B, d] representing x_t at each time step
        x_hat_trajectory: List of tensors, each of shape [B, d] representing x̂₀ predictions
        M_trajectory: List of tensors, each of shape [B, d] representing M_t at each time step
        x0_target: Tensor of shape [B, d] - target x0 values
        x1_batch: Tensor of shape [B, d] - starting x1 values  
        mode: String indicating bridge type
        K: Number of time steps
        title: Plot title
    """
    # Always create 3x4 subplot grid for x_t, x̂₀, and M_t
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    # Convert trajectory data to numpy arrays
    # trajectory: list of [B, d] tensors -> [K+1, B, d] array
    traj_array = np.array([step.cpu().numpy() for step in trajectory])  # [K+1, B, d]
    x_hat_array = np.array([step.cpu().numpy() for step in x_hat_trajectory])  # [K, B, d]
    M_array = np.array([step.cpu().numpy() for step in M_trajectory])  # [K, B, d]
    
    B, d = x0_target.shape
    time_steps = np.linspace(0.0, 1.0, len(traj_array))  # Reverse time from 1 to 0
    x_hat_time_steps = np.linspace(0.0, 1.0, len(x_hat_array))  # Match x_hat_array length
    M_time_steps = np.linspace(0.0, 1.0, len(M_array))  # Match M_array length
    
    # Plot first 4 dimensions
    n_dims_to_plot = min(4, d)
    
    # Top row: x_t trajectories
    for dim in range(n_dims_to_plot):
        ax = axes[0, dim]
        
        # Plot all trajectories with low alpha for density visualization
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.05 if B > 100 else 0.1
            ax.plot(time_steps, traj_array[:, b, dim], 'b-', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for summary statistics
        traj_percentiles = np.percentile(traj_array[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(time_steps, traj_percentiles[2], 'r-', linewidth=2, label='Median x_t')
        ax.fill_between(time_steps, traj_percentiles[1], traj_percentiles[3], 
                       alpha=0.4, color='red', label='25-75%')
        ax.fill_between(time_steps, traj_percentiles[0], traj_percentiles[4], 
                       alpha=0.3, color='red', label='10-90%')
        
        # Plot target mean as reference
        target_mean = x0_target[:, dim].float().mean().cpu().numpy()
        x1_mean = x1_batch[:, dim].float().mean().cpu().numpy()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
        
        ax.set_title(f'x_t Trajectories - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
        # Don't invert x-axis since time_steps already goes from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Second row: x̂₀ predictions
    for dim in range(n_dims_to_plot):
        ax = axes[1, dim]
        
        # Plot all x̂₀ predictions with low alpha
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.05 if B > 100 else 0.1
            ax.plot(x_hat_time_steps, x_hat_array[:, b, dim], 'purple', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for x̂₀ predictions
        xhat_percentiles = np.percentile(x_hat_array[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(x_hat_time_steps, xhat_percentiles[2], 'purple', linewidth=2, label='Median x̂₀')
        ax.fill_between(x_hat_time_steps, xhat_percentiles[1], xhat_percentiles[3], 
                       alpha=0.4, color='purple', label='25-75%')
        ax.fill_between(x_hat_time_steps, xhat_percentiles[0], xhat_percentiles[4], 
                       alpha=0.3, color='purple', label='10-90%')
        
        # Plot target mean as reference
        target_mean = x0_target[:, dim].float().mean().cpu().numpy()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        
        ax.set_title(f'x̂₀ Predictions - Dimension {dim}')
        ax.set_xlabel('Time t (1→0)')
        ax.set_ylabel('Predicted x₀')
        ax.grid(True, alpha=0.3)
        # Don't invert x-axis since x_hat_time_steps already goes from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Third row: M_t trajectories
    for dim in range(n_dims_to_plot):
        ax = axes[2, dim]
        
        # Plot all M_t trajectories with low alpha
        for b in range(min(B, 1000)):  # Limit to 1000 trajectories for performance
            alpha = 0.05 if B > 100 else 0.1
            ax.plot(M_time_steps, M_array[:, b, dim], 'orange', alpha=alpha, linewidth=0.5)
        
        # Plot percentiles for M_t
        M_percentiles = np.percentile(M_array[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(M_time_steps, M_percentiles[2], 'orange', linewidth=2, label='Median M_t')
        ax.fill_between(M_time_steps, M_percentiles[1], M_percentiles[3], 
                       alpha=0.4, color='orange', label='25-75%')
        ax.fill_between(M_time_steps, M_percentiles[0], M_percentiles[4], 
                       alpha=0.3, color='orange', label='10-90%')
        
        ax.set_title(f'M_t Trajectories - Dimension {dim}')
        ax.set_xlabel('Time t (1→0)')
        ax.set_ylabel('M_t value')
        ax.grid(True, alpha=0.3)
        # Don't invert x-axis since M_time_steps already goes from 1 to 0
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    plt.tight_layout()
    return fig


def plot_true_marginal_distributions(x0_batch, x1_batch, z_batch, bridge_collate, times=None, 
                                    n_samples=1000, title="True Bridge Marginals"):
    """
    Plot true x_t and M_t marginal distributions as line plots over time.
    Format exactly matches plot_full_reverse_trajectories for direct comparison.
    
    Args:
        x0_batch: Tensor of shape [B, d] - x0 samples
        x1_batch: Tensor of shape [B, d] - x1 samples  
        z_batch: Tensor of shape [B, z_dim] - context
        bridge_collate: Bridge collate object (e.g., PoissonBDBridgeCollate)
        times: List of time points to sample at (if None, uses dense sampling)
        n_samples: Number of samples to generate for each time point
        title: Plot title
    """
    # Use dense time sampling for smooth line plots (like reverse trajectories)
    if times is None:
        times = np.linspace(0.0, 1.0, 21)  # Dense sampling: 21 points from 0 to 1
    else:
        times = np.array(times)
    
    # Match the exact layout of plot_full_reverse_trajectories
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    d = x0_batch.shape[1]
    n_dims_to_plot = min(4, d)
    
    # Collect samples for all time points
    print(f"Sampling bridge marginals at {len(times)} time points...")
    all_x_t_samples = []  # List of [n_samples, d] tensors
    all_M_samples = []    # List of [n_samples, d] tensors
    
    for i, t_val in enumerate(times):
        if i % 5 == 0:  # Progress indicator
            print(f"  Progress: {i+1}/{len(times)} (t={t_val:.2f})")
    
        # Create batch data for this time point
        batch_data = []
        for j in range(n_samples):
            idx = j % len(x0_batch)  # Cycle through available data
            batch_data.append({
                'x0': x0_batch[idx],
                'x1': x1_batch[idx], 
                'z': z_batch[idx]
            })
        
        # Use the bridge directly with specified time
        result = bridge_collate(batch_data, t_target=t_val)
        
        # Extract x_t and M_t
        x_t = result['x_t']
        M_t = result.get('M', torch.zeros_like(x_t))  # Some bridges might not have M
        
        # Store samples
        all_x_t_samples.append(x_t.cpu())
        all_M_samples.append(M_t.cpu())
                
    
    # Convert to arrays for plotting: [time_steps, n_samples, d]
    x_t_array = np.array([step.numpy() for step in all_x_t_samples])  # [T, B, d]
    M_array = np.array([step.numpy() for step in all_M_samples])      # [T, B, d]
    
    print(f"Collected samples shape: x_t={x_t_array.shape}, M_t={M_array.shape}")
    
    # Top row: x_t evolution over time (exactly like x_t trajectories)
    for dim in range(n_dims_to_plot):
        ax = axes[0, dim]
        
        # Plot percentiles over time (exactly like reverse trajectories)
        traj_percentiles = np.percentile(x_t_array[:, :, dim], [10, 25, 50, 75, 90], axis=1)
        ax.plot(times, traj_percentiles[2], 'r-', linewidth=2, label='Median x_t')
        ax.fill_between(times, traj_percentiles[1], traj_percentiles[3], 
                       alpha=0.4, color='red', label='25-75%')
        ax.fill_between(times, traj_percentiles[0], traj_percentiles[4], 
                       alpha=0.3, color='red', label='10-90%')
        
        # Plot target mean as reference
        target_mean = x0_batch[:, dim].float().mean().cpu().numpy()
        x1_mean = x1_batch[:, dim].float().mean().cpu().numpy()
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        ax.axhline(y=x1_mean, color='orange', linestyle='--', linewidth=2, label='x₁ mean')
        
        ax.set_title(f'x_t Evolution - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Middle row: Linear interpolation reference (like x̂₀ predictions)
    for dim in range(n_dims_to_plot):
        ax = axes[1, dim]
        
        # Plot linear interpolation between x0 and x1 means
        target_mean = x0_batch[:, dim].float().mean().cpu().numpy()
        x1_mean = x1_batch[:, dim].float().mean().cpu().numpy()
        linear_interp = (1 - times) * target_mean + times * x1_mean
        
        ax.plot(times, linear_interp, 'purple', linewidth=2, label='Linear interpolation')
        ax.axhline(y=target_mean, color='green', linestyle='--', linewidth=2, label='Target x₀ mean')
        
        ax.set_title(f'Reference Interpolation - Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Expected value')
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize='small', loc='best')
    
    # Bottom row: M_t evolution over time (exactly like M_t trajectories)
    for dim in range(n_dims_to_plot):
        ax = axes[2, dim]
        
        # Check if we have meaningful M_t data
        if M_array.max() > 0:
            # Plot percentiles over time (exactly like reverse trajectories)
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
        else:
            # If M_t not available, show placeholder
            ax.text(0.5, 0.5, f'M_t not available\nfor this bridge type', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'M_t Evolution - Dimension {dim}')
    
    plt.tight_layout()
    return fig