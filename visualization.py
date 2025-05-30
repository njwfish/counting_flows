"""
Visualization Functions for Count-based Flow Matching

Provides comprehensive plotting and debugging visualization for count flows.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from .scheduling import make_time_spacing_schedule, make_r_schedule
from .samplers import reverse_sampler


def plot_schedule_comparison(K=50, title="Schedule Comparison"):
    """Visualize different r(t) schedule types"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    # R schedules (for NB bridge)
    schedule_types = ["linear", "cosine", "exponential", "polynomial", "sqrt", "sigmoid"]
    
    for i, schedule_type in enumerate(schedule_types):
        ax = axes[i//3, i%3]
        
        try:
            # Default parameters for each schedule
            kwargs = {}
            if schedule_type == "exponential":
                kwargs = {'decay_rate': 2.0}
            elif schedule_type == "polynomial":
                kwargs = {'power': 2.0}
            elif schedule_type == "sigmoid":
                kwargs = {'steepness': 10.0, 'midpoint': 0.5}
            
            r_sched = make_r_schedule(K, r_min=1.0, r_max=20.0, 
                                    r_schedule=schedule_type, **kwargs)
            times = torch.linspace(0, 1, K)
            
            ax.plot(times.numpy(), r_sched.numpy(), linewidth=2, label=schedule_type)
            ax.set_title(f'{schedule_type.title()} Schedule')
            ax.set_xlabel('Time t')
            ax.set_ylabel('r(t) value')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 21)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center')
            ax.set_title(f'{schedule_type.title()} (Error)')
    
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


def plot_reverse_trajectory(x1, z, model, K=30, mode="poisson", device="cuda", n_trajectories=5, 
                           r_min=1.0, r_max=20.0, r_schedule="linear", time_schedule="uniform", **schedule_kwargs):
    """Plot the reverse sampling trajectory"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Reverse Sampling Trajectories ({mode} bridge)")
    
    # Get the time schedule used in sampling
    time_points = make_time_spacing_schedule(K, time_schedule, **schedule_kwargs)
    time_points = torch.flip(time_points, [0])  # Reverse for backward process
    
    trajectories = []
    
    for traj in range(n_trajectories):
        # Use the actual reverse sampler to get full trajectory
        x_start = x1[:1].clone().to(device)  # Single sample
        z_start = z[:1].clone().to(device)
        
        # Get the full reverse trajectory with intermediate steps
        x_final, trajectory = reverse_sampler(
            x_start, z_start, model,
            K=K, mode=mode, device=device,
            r_min=r_min, r_max=r_max,
            r_schedule=r_schedule, time_schedule=time_schedule,
            return_trajectory=True,
            **schedule_kwargs
        )
        
        # Convert trajectory to numpy array for plotting
        # trajectory is a list of tensors, each of shape [1, d]
        traj_array = np.array([step[0].numpy() for step in trajectory])  # Shape: [K+1, d]
        
        trajectories.append(traj_array)
    
    # Convert time points to numpy for plotting
    time_np = time_points.numpy()
    
    # Plot trajectories for each dimension
    for dim in range(min(4, x1.shape[1])):
        ax = axes[dim//2, dim%2]
        
        # Plot linear interpolant reference (dotted black line)
        x1_val = x1[0, dim].cpu().numpy()
        x0_mean = np.mean([traj[-1, dim] for traj in trajectories])  # Average final point
        linear_interp = np.linspace(x1_val, x0_mean, len(time_np))
        ax.plot(time_np, linear_interp, 'k--', alpha=0.5, linewidth=2, label='Linear interpolant')
        
        # Plot actual trajectories
        for traj in trajectories:
            ax.plot(time_np, traj[:, dim], 'o-', alpha=0.7, linewidth=2, markersize=4)
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Invert x-axis since we're going backwards in time (t=1 → t=0)
        ax.invert_xaxis()
    
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