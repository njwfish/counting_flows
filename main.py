"""
Main script for Count-based Flow Matching

Coordinates training, generation, and visualization of count-based flow models.
"""

import torch
from torch.distributions import Poisson
import matplotlib.pyplot as plt
import os
from pathlib import Path

from .cli import parse_args
from .models import NBPosterior, BetaBinomialPosterior, MLERegressor, ZeroInflatedPoissonPosterior
from .datasets import PoissonBridgeCollate, NBBridgeCollate
from .training import train_model, create_training_dataloader
from .samplers import reverse_sampler
from .visualization import (
    plot_schedule_comparison, plot_time_spacing_comparison,
    plot_reverse_trajectory, plot_generation_analysis, plot_loss_curve
)


def setup_plot_directories(base_dir="plots"):
    """Create organized directory structure for plots"""
    dirs = [
        f"{base_dir}/schedules",
        f"{base_dir}/training", 
        f"{base_dir}/sampling",
        f"{base_dir}/analysis"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return {
        'schedules': f"{base_dir}/schedules",
        'training': f"{base_dir}/training",
        'sampling': f"{base_dir}/sampling",
        'analysis': f"{base_dir}/analysis"
    }


def main():
    """Main entry point for the counting flows framework"""
    args = parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Quick mode adjustments
    if args.quick:
        args.iterations = 2000
        args.gen_samples = 1000
    
    # Set sampling schedules
    sample_bridge_mode = args.sample_bridge if args.sample_bridge != "auto" else args.bridge
    sample_r_schedule = args.sample_r_schedule if args.sample_r_schedule != "auto" else args.r_schedule
    sample_time_schedule = args.sample_time_schedule if args.sample_time_schedule != "auto" else args.time_schedule
    
    # Collect schedule kwargs
    schedule_kwargs = {
        'decay_rate': args.decay_rate,
        'steepness': args.steepness,
        'midpoint': args.midpoint,
        'power': args.power,
        'concentration': args.concentration
    }
    
    # Setup plot directories
    plot_dirs = setup_plot_directories() if args.plot else None
    
    print("Count-based Flow Matching")
    print("=" * 50)
    print(f"Dataset:             {args.dataset}")
    print(f"Architecture:        {args.arch}")
    print(f"Training bridge:     {args.bridge}")
    print(f"Sampling bridge:     {sample_bridge_mode}")
    print(f"R schedule:          {args.r_schedule}")
    print(f"Time schedule:       {args.time_schedule}")
    print(f"Sample R schedule:   {sample_r_schedule}")
    print(f"Sample time sched:   {sample_time_schedule}")
    print(f"Device:              {device}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Steps:               {args.steps}")
    print(f"Iterations:          {args.iterations}")
    if args.quick:
        print("Mode:                QUICK")
    print()
    
    # Determine context dimension based on dataset type
    if args.dataset == "poisson":
        context_dim = args.data_dim * 2  # [lam0, lam1]
    elif args.dataset == "betabinomial":
        context_dim = args.data_dim * 6  # [n0, alpha0, beta0, n1, alpha1, beta1]
    else:
        context_dim = args.data_dim * 2  # Default
    
    # Create model
    if args.arch == "nb":
        model = NBPosterior(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: Negative Binomial posterior")
    elif args.arch == "bb":
        model = BetaBinomialPosterior(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: Beta-Binomial posterior")
    elif args.arch == "zip":
        model = ZeroInflatedPoissonPosterior(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: Zero-Inflated Poisson posterior")
    else:
        model = MLERegressor(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: MLE regressor")
    
    # Create training DataLoader based on bridge mode
    dataset_size = args.dataset_size if args.dataset_size is not None else max(args.iterations, 10000)
    
    base_mode = "fixed base" if args.fixed_base else "random base"
    print(f"Dataset: {args.dataset.title()} ({args.data_dim}D, {base_mode})")
    
    if args.bridge == "nb":
        train_dataloader, train_dataset = create_training_dataloader(
            bridge_type="nb",
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            d=args.data_dim,
            n_steps=args.steps,
            dataset_size=dataset_size,
            fixed_base=args.fixed_base,
            r_min=args.r_min,
            r_max=args.r_max,
            r_schedule=args.r_schedule,
            time_schedule=args.time_schedule,
            **schedule_kwargs
        )
        print(f"Bridge: Negative Binomial (Polya) for training")
        print(f"  R schedule: {args.r_schedule}")
        print(f"  Time schedule: {args.time_schedule}")
    elif args.bridge == "poisson":
        train_dataloader, train_dataset = create_training_dataloader(
            bridge_type="poisson",
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            d=args.data_dim,
            n_steps=args.steps,
            dataset_size=dataset_size,
            fixed_base=args.fixed_base,
            time_schedule=args.time_schedule,
            **schedule_kwargs
        )
        print(f"Bridge: Poisson for training")
        print(f"  Time schedule: {args.time_schedule}")
    
    # Train
    print(f"\nTraining model...")
    model, losses = train_model(
        model, train_dataloader,
        num_iterations=args.iterations,
        lr=args.lr, 
        device=device
    )
    
    # Generate samples
    print(f"\nGenerating {args.gen_samples} samples...")
    print(f"Using {sample_bridge_mode} bridge for sampling")
    print(f"R schedule: {sample_r_schedule}")
    print(f"Time schedule: {sample_time_schedule}")
    
    with torch.no_grad():
        # Sample evaluation batch directly from dataset
        print("Sampling evaluation batch from dataset...")
        
        # Create dataset that generates batches directly
        eval_dataset = train_dataset.__class__(
            size=args.gen_samples,  # Total size
            d=args.data_dim,
            fixed_base=args.fixed_base,
            batch_size=args.gen_samples,  # Generate one large batch
            homogeneous=False,  # Different parameters per sample
            **{k: v for k, v in vars(train_dataset).items() 
               if k not in ['size', 'd', 'fixed_base', 'batch_size', 'homogeneous', 'base_params']}
        )
        
        # Get the batch (x0_target, x1, z)
        batch = eval_dataset[0]  # Single batch containing all samples
        x0_target = batch['x0'].to(device)
        x1_batch = batch['x1'].to(device)
        z_batch = batch['z'].to(device)
        
        # Use reverse sampler to generate x0 from x1 using the bridge
        print(f"Using reverse sampler to generate x0 from x1...")
        x0_samples = reverse_sampler(
            x1_batch, z_batch, model,
            K=args.steps,
            mode=sample_bridge_mode,
            r_min=args.r_min,
            r_max=args.r_max,
            r_schedule=sample_r_schedule,
            time_schedule=sample_time_schedule,
            use_mean=args.use_mean,
            device=device,
            **schedule_kwargs
        )
        
        # Compute statistics
        sample_mean = x0_samples.float().mean(0)
        target_mean = x0_target.float().mean(0) 
        sample_std = x0_samples.float().std(0)
        target_std = x0_target.float().std(0)
        
        print(f"\nResults:")
        print(f"Generated samples shape: {x0_samples.shape}")
        print(f"Sample mean:  {sample_mean.cpu().numpy()}")
        print(f"Target mean:  {target_mean.cpu().numpy()}")
        print(f"Sample std:   {sample_std.cpu().numpy()}")
        print(f"Target std:   {target_std.cpu().numpy()}")
        
        # Compute error metrics
        mean_error = torch.abs(sample_mean - target_mean).mean()
        std_error = torch.abs(sample_std - target_std).mean()
        print(f"Mean abs error (mean): {mean_error:.4f}")
        print(f"Mean abs error (std):  {std_error:.4f}")

    if args.plot:
        print(f"\nGenerating diagnostic plots...")
        
        # Create config string for filenames
        config_str = f"{args.arch}_{args.bridge}_{args.r_schedule}_{args.time_schedule}"
        sample_config_str = f"{args.arch}_{sample_bridge_mode}_{sample_r_schedule}_{sample_time_schedule}"
        
        # Plot schedule comparisons first
        fig = plot_schedule_comparison(K=args.steps, title="R(t) Schedule Comparison")
        plt.savefig(f"{plot_dirs['schedules']}/r_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        fig = plot_time_spacing_comparison(K=args.steps, title="Time Spacing Comparison") 
        plt.savefig(f"{plot_dirs['schedules']}/time_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot losses
        fig = plot_loss_curve(losses, f"Training Loss ({args.arch} + {args.bridge} bridge)")
        plt.savefig(f"{plot_dirs['training']}/loss_{config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot reverse trajectory
        fig = plot_reverse_trajectory(
            x1_batch, z_batch, model, 
            K=args.steps, 
            mode=sample_bridge_mode, 
            device=device,
            n_trajectories=5,
            r_min=args.r_min,
            r_max=args.r_max,
            r_schedule=sample_r_schedule,
            time_schedule=sample_time_schedule,
            **schedule_kwargs
        )
        plt.savefig(f"{plot_dirs['sampling']}/trajectories_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot generation analysis
        fig = plot_generation_analysis(x0_samples, x0_target, x1_batch, 
                                     title=f"Generation Analysis ({args.arch} + {sample_bridge_mode}, {sample_r_schedule}, {sample_time_schedule})")
        plt.savefig(f"{plot_dirs['analysis']}/generation_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to organized directories:")
        print(f"  Schedules: {plot_dirs['schedules']}/")
        print(f"  Training:  {plot_dirs['training']}/")
        print(f"  Sampling:  {plot_dirs['sampling']}/")
        print(f"  Analysis:  {plot_dirs['analysis']}/")
        print(f"\nNote: Some debug plots were removed due to deprecated code dependencies.")


if __name__ == "__main__":
    main() 