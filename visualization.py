"""
Visualization Functions for Count-based Flow Matching

Provides comprehensive plotting and debugging visualization for count flows.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Binomial, Beta
from .scheduling import make_time_schedule, make_time_spacing_schedule
from .models import NBPosterior, BetaBinomialPosterior, ZeroInflatedPoissonPosterior


def plot_schedule_comparison(K=50, title="Schedule Comparison"):
    """Visualize different schedule types"""
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
            
            times, weights = make_time_schedule(
                K, schedule_type=schedule_type, 
                start_val=20.0, end_val=1.0, 
                **kwargs
            )
            
            ax.plot(times.numpy(), weights.numpy(), linewidth=2, label=schedule_type)
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


def plot_bridge_dynamics(sampler, B=1000, d=4, n_steps=30, title="Bridge Dynamics"):
    """Visualize how the bridge evolves over time"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    
    # Collect data at different time points
    time_points = [0.2, 0.4, 0.6, 0.8]
    all_data = {t: [] for t in time_points}
    
    for _ in range(10):  # Multiple samples for statistics
        x0, x1, x_t, ts, z, r = sampler(B, d, n_steps)
        
        for i, target_t in enumerate(time_points):
            # Find samples close to this time point
            mask = torch.abs(ts.squeeze() - target_t) < 0.05
            if mask.sum() > 0:
                all_data[target_t].append(x_t[mask].numpy())
    
    for i, t in enumerate(time_points):
        ax = axes[i//2, i%2]
        if all_data[t]:
            data = np.concatenate(all_data[t], axis=0)
            ax.hist(data[:, 0], bins=30, alpha=0.7, density=True, label=f't={t}')
            ax.set_title(f'x_t distribution at t={t}')
            ax.set_xlabel('Count value')
            ax.set_ylabel('Density')
    
    plt.tight_layout()
    return fig


def plot_reverse_trajectory(x1, z, model, K=30, mode="poisson", device="cuda", n_trajectories=5):
    """Plot the reverse sampling trajectory"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Reverse Sampling Trajectories ({mode} bridge)")
    
    trajectories = []
    
    for traj in range(n_trajectories):
        x = x1[:1].clone().to(device).float()  # Single sample
        dt = 1.0 / K
        
        traj_data = [x.cpu().numpy().copy()]
        
        for k in range(K, 0, -1):
            t = torch.full_like(x[:, :1], k * dt)
            x0_hat = model.sample(x, z[:1], t, use_mean=False).float()
            
            delta = x - x0_hat
            n = delta.abs().clamp_max(1e6).long()
            sgn = torch.sign(delta)
            
            if mode == "nb":
                r_k = 1.0  # Simplified for visualization
                alpha = max(r_k * (k-1) * dt, 1e-6)
                beta = max(r_k * dt, 1e-6)
                p_step = Beta(alpha, beta).sample(n.shape).clamp(0., 0.999).to(device)
            else:
                p_step = torch.full_like(x, dt).clamp(0., 0.999)
            
            krem = torch.zeros_like(n).float().to(device)
            mask = n > 0
            
            if mask.any():
                krem[mask] = Binomial(
                    total_count=n[mask],
                    probs=p_step[mask]
                ).sample()
            
            x = x - sgn * krem
            traj_data.append(x.cpu().numpy().copy())
        
        trajectories.append(np.array(traj_data))
    
    # Plot trajectories for each dimension
    for dim in range(min(4, x1.shape[1])):
        ax = axes[dim//2, dim%2]
        for traj in trajectories:
            steps = np.arange(len(traj))
            ax.plot(steps, traj[:, 0, dim], alpha=0.7, linewidth=2)
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Reverse step')
        ax.set_ylabel('Count value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_generation_analysis(x0_samples, lam0_batch, lam1_batch, title="Generation Analysis"):
    """Analyze the quality of generated samples"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    d = min(4, x0_samples.shape[1])
    
    # Plot histograms for each dimension
    for dim in range(min(2, d)):
        ax = axes[0, dim]
        
        samples = x0_samples[:, dim].cpu().numpy()
        target_lambda = lam0_batch[0, dim].cpu().numpy()
        
        # Generate theoretical Poisson samples for comparison
        theoretical = np.random.poisson(target_lambda, len(samples))
        
        ax.hist(samples, bins=30, alpha=0.7, density=True, label='Generated', color='blue')
        ax.hist(theoretical, bins=30, alpha=0.5, density=True, label=f'Poisson(λ={target_lambda:.2f})', color='red')
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Count value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # Sample statistics comparison
    ax = axes[0, 2]
    sample_means = x0_samples.float().mean(0).cpu().numpy()
    target_means = lam0_batch[0].cpu().numpy()
    
    dims = np.arange(len(sample_means))
    width = 0.35
    ax.bar(dims - width/2, sample_means, width, label='Generated mean', alpha=0.7)
    ax.bar(dims + width/2, target_means, width, label='Target λ₀', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean count')
    ax.set_title('Mean Comparison')
    ax.legend()
    
    # Variance comparison
    ax = axes[1, 0]
    sample_vars = x0_samples.float().var(0).cpu().numpy()
    target_vars = lam0_batch[0].cpu().numpy()  # For Poisson, var = mean
    
    ax.bar(dims - width/2, sample_vars, width, label='Generated var', alpha=0.7)
    ax.bar(dims + width/2, target_vars, width, label='Target var (λ₀)', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Comparison')
    ax.legend()
    
    # Scatter plot: generated vs target means
    ax = axes[1, 1]
    ax.scatter(target_means, sample_means, alpha=0.7)
    min_val, max_val = min(target_means.min(), sample_means.min()), max(target_means.max(), sample_means.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect match')
    ax.set_xlabel('Target λ₀')
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


def debug_model_predictions(model, sampler, device="cuda", n_samples=100):
    """Debug what the model is actually predicting"""
    model.eval()
    
    # Sample some data
    x0, x1, x_t, ts, z, r = [
        t.to(device) if torch.is_tensor(t) else t
        for t in sampler(n_samples, 4, 30)
    ]
    
    with torch.no_grad():
        if isinstance(model, NBPosterior):
            raw_output = model.net(torch.cat([x_t.float(), ts.float(), z.float()], dim=-1))
            r_pred, p_pred = model.process_output(raw_output)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("NB Model Predictions Debug")
            
            # Plot r predictions
            axes[0, 0].hist(r_pred.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 0].set_title('Predicted r values')
            axes[0, 0].set_xlabel('r')
            
            # Plot p predictions
            axes[0, 1].hist(p_pred.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 1].set_title('Predicted p values')
            axes[0, 1].set_xlabel('p')
            
            # Plot NB means
            nb_means = r_pred * (1 - p_pred) / p_pred
            axes[0, 2].hist(nb_means.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 2].set_title('NB Distribution Means')
            axes[0, 2].set_xlabel('Mean')
            
            # Compare with true x0
            axes[1, 0].scatter(x0.cpu().numpy().flatten(), nb_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 0].set_xlabel('True x₀')
            axes[1, 0].set_ylabel('Predicted mean')
            axes[1, 0].set_title('Prediction vs Truth')
            
            # Plot time vs predictions
            ts_expanded = ts.expand_as(nb_means)
            axes[1, 1].scatter(ts_expanded.cpu().numpy().flatten(), nb_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 1].set_xlabel('Time t')
            axes[1, 1].set_ylabel('Predicted mean')
            axes[1, 1].set_title('Time vs Predictions')
            
            # Plot residuals
            residuals = x0.float() - nb_means
            axes[1, 2].hist(residuals.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[1, 2].set_title('Prediction Residuals')
            axes[1, 2].set_xlabel('x₀ - predicted_mean')
            
        elif isinstance(model, BetaBinomialPosterior):
            raw_output = model.net(torch.cat([x_t.float(), ts.float(), z.float()], dim=-1))
            n_pred, alpha_pred, beta_pred = model.process_output(raw_output)
            
            concentration = alpha_pred + beta_pred
            mean_p = alpha_pred / concentration
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("Beta-Binomial Model Predictions Debug")
            
            # Plot n predictions
            axes[0, 0].hist(n_pred.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 0].set_title('Predicted n values')
            axes[0, 0].set_xlabel('n')
            
            # Plot mean_p predictions
            axes[0, 1].hist(mean_p.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 1].set_title('Predicted mean_p values')
            axes[0, 1].set_xlabel('mean_p = α/(α+β)')
            
            # Plot concentration predictions
            axes[0, 2].hist(concentration.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 2].set_title('Predicted concentration values')
            axes[0, 2].set_xlabel('concentration = α+β')
            
            # Plot BB means
            bb_means = n_pred * alpha_pred / (alpha_pred + beta_pred)
            axes[1, 0].scatter(x0.cpu().numpy().flatten(), bb_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 0].set_xlabel('True x₀')
            axes[1, 0].set_ylabel('Predicted mean')
            axes[1, 0].set_title('Prediction vs Truth')
            
            # Plot time vs predictions
            ts_expanded = ts.expand_as(bb_means)
            axes[1, 1].scatter(ts_expanded.cpu().numpy().flatten(), bb_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 1].set_xlabel('Time t')
            axes[1, 1].set_ylabel('Predicted mean')
            axes[1, 1].set_title('Time vs Predictions')
            
            # Plot residuals
            residuals = x0.float() - bb_means
            axes[1, 2].hist(residuals.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[1, 2].set_title('Prediction Residuals')
            axes[1, 2].set_xlabel('x₀ - predicted_mean')

        elif isinstance(model, ZeroInflatedPoissonPosterior):
            raw_output = model.net(torch.cat([x_t.float(), ts.float(), z.float()], dim=-1))
            lam_pred, pi_pred = model.process_output(raw_output)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("Zero-Inflated Poisson Model Predictions Debug")
            
            # Plot lambda predictions
            axes[0, 0].hist(lam_pred.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 0].set_title('Predicted λ values')
            axes[0, 0].set_xlabel('λ')
            
            # Plot pi predictions
            axes[0, 1].hist(pi_pred.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 1].set_title('Predicted π values (zero inflation)')
            axes[0, 1].set_xlabel('π')
            
            # Plot ZIP means
            zip_means = (1 - pi_pred) * lam_pred
            axes[0, 2].hist(zip_means.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[0, 2].set_title('ZIP Distribution Means')
            axes[0, 2].set_xlabel('Mean = (1-π)λ')
            
            # Compare with true x0
            axes[1, 0].scatter(x0.cpu().numpy().flatten(), zip_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 0].set_xlabel('True x₀')
            axes[1, 0].set_ylabel('Predicted mean')
            axes[1, 0].set_title('Prediction vs Truth')
            
            # Plot time vs predictions
            ts_expanded = ts.expand_as(zip_means)
            axes[1, 1].scatter(ts_expanded.cpu().numpy().flatten(), zip_means.cpu().numpy().flatten(), alpha=0.5)
            axes[1, 1].set_xlabel('Time t')
            axes[1, 1].set_ylabel('Predicted mean')
            axes[1, 1].set_title('Time vs Predictions')
            
            # Plot residuals
            residuals = x0.float() - zip_means
            axes[1, 2].hist(residuals.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[1, 2].set_title('Prediction Residuals')
            axes[1, 2].set_xlabel('x₀ - predicted_mean')

        else:  # MLERegressor
            log_pred = model(x_t, z, ts)
            x0_pred = torch.exp(log_pred) - 1
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle("MLE Model Predictions Debug")
            
            axes[0].scatter(x0.cpu().numpy().flatten(), x0_pred.cpu().numpy().flatten(), alpha=0.5)
            axes[0].set_xlabel('True x₀')
            axes[0].set_ylabel('Predicted x₀')
            axes[0].set_title('Prediction vs Truth')
            
            ts_expanded = ts.expand_as(x0_pred)
            axes[1].scatter(ts_expanded.cpu().numpy().flatten(), x0_pred.cpu().numpy().flatten(), alpha=0.5)
            axes[1].set_xlabel('Time t')
            axes[1].set_ylabel('Predicted x₀')
            axes[1].set_title('Time vs Predictions')
            
            residuals = x0.float() - x0_pred
            axes[2].hist(residuals.cpu().numpy().flatten(), bins=30, alpha=0.7)
            axes[2].set_title('Prediction Residuals')
            axes[2].set_xlabel('x₀ - predicted')
    
    plt.tight_layout()
    return fig 