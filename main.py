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
from .datasets import create_dataloader
from .samplers import reverse_sampler
from .training import train_model, create_training_dataloader
from .visualization import (
    plot_schedule_comparison, plot_time_spacing_comparison, plot_bridge_dynamics,
    plot_reverse_trajectory, plot_generation_analysis, plot_loss_curve, debug_model_predictions
)

# Legacy bridge samplers for visualization (still needed for plotting functions)
# from .bridges import sample_batch_poisson, sample_batch_nb


def setup_plot_directories(base_dir="plots"):
    """Create organized directory structure for plots"""
    dirs = [
        f"{base_dir}/schedules",
        f"{base_dir}/training", 
        f"{base_dir}/bridges",
        f"{base_dir}/sampling",
        f"{base_dir}/analysis"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return {
        'schedules': f"{base_dir}/schedules",
        'training': f"{base_dir}/training",
        'bridges': f"{base_dir}/bridges", 
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
    plot_dirs = setup_plot_directories() if args.debug else None
    
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
            lam_scale=args.data_scale,
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
            lam_scale=args.data_scale,
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
        # Generate target distributions based on dataset type
        if args.dataset == "poisson":
            if args.fixed_base:
                # Fixed λ₀, random λ₁
                lam0 = torch.full((1, args.data_dim), 5.0)  # Use default fixed lambda
                lam1 = args.data_scale * torch.rand(1, args.data_dim)
            else:
                # Both random
                lam0 = args.data_scale * torch.rand(1, args.data_dim)
                lam1 = args.data_scale * torch.rand(1, args.data_dim)
            
            target_std_0 = torch.sqrt(lam0)  # Poisson std = sqrt(λ)
            target_std_1 = torch.sqrt(lam1)
            
        elif args.dataset == "betabinomial":
            # BetaBinomial is more complex - approximate with means for sampling
            if args.fixed_base:
                # Fixed parameters for x₀
                n0 = torch.full((1, args.data_dim), 10.0)
                alpha0 = torch.full((1, args.data_dim), 0.5)
                beta0 = torch.full((1, args.data_dim), 0.5)
                # Random parameters for x₁
                n1 = torch.randint(5, 21, (1, args.data_dim)).float()
                alpha1 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
                beta1 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
            else:
                # Both random
                n0 = torch.randint(5, 21, (1, args.data_dim)).float()
                alpha0 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
                beta0 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
                n1 = torch.randint(5, 21, (1, args.data_dim)).float()
                alpha1 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
                beta1 = 0.1 + 1.9 * torch.rand(1, args.data_dim)
            
            # BetaBinomial mean = n * alpha / (alpha + beta)
            # BetaBinomial var = n * alpha * beta * (alpha + beta + n) / ((alpha + beta)^2 * (alpha + beta + 1))
            mean0 = n0 * alpha0 / (alpha0 + beta0)
            mean1 = n1 * alpha1 / (alpha1 + beta1)
            
            var0 = n0 * alpha0 * beta0 * (alpha0 + beta0 + n0) / ((alpha0 + beta0)**2 * (alpha0 + beta0 + 1))
            var1 = n1 * alpha1 * beta1 * (alpha1 + beta1 + n1) / ((alpha1 + beta1)**2 * (alpha1 + beta1 + 1))
            
            target_std_0 = torch.sqrt(var0)
            target_std_1 = torch.sqrt(var1)
            
            # For reverse sampling, approximate with Poisson using means
            lam0 = mean0
            lam1 = mean1
            
        else:
            # Default fallback
            lam0 = args.data_scale * torch.rand(1, args.data_dim)
            lam1 = args.data_scale * torch.rand(1, args.data_dim)
            target_std_0 = torch.sqrt(lam0)
            target_std_1 = torch.sqrt(lam1)
        
        lam1_batch = lam1.repeat(args.gen_samples, 1)
        lam0_batch = lam0.repeat(args.gen_samples, 1)
        
        x1 = Poisson(lam1_batch).sample().to(device)
        z_gen = torch.cat([lam0_batch, lam1_batch], dim=1).to(device)
        
        x0_samples = reverse_sampler(
            x1, z_gen, model,
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
        
        print(f"\nResults:")
        print(f"Generated samples shape: {x0_samples.shape}")
        print(f"Sample mean:  {x0_samples.float().mean(0).cpu().numpy()}")
        print(f"Target λ₀:    {lam0_batch[0].cpu().numpy()}")
        print(f"Target λ₁:    {lam1_batch[0].cpu().numpy()}")
        print(f"Sample std:   {x0_samples.float().std(0).cpu().numpy()}")
        print(f"Target std₀:  {target_std_0[0].cpu().numpy()}")
        print(f"Target std₁:  {target_std_1[0].cpu().numpy()}")

    if args.debug:
        print(f"\nGenerating diagnostic plots...")
        
        # Create mock legacy samplers for visualization functions
        # These generate data in the old format that the plotting functions expect
        def create_mock_legacy_sampler(bridge_type, dataset_type):
            def mock_sampler(B, d, n_steps):
                # Create a small batch using the new system
                temp_dataloader, _ = create_training_dataloader(
                    bridge_type=bridge_type,
                    dataset_type=dataset_type,
                    batch_size=B,
                    d=d,
                    n_steps=n_steps,
                    dataset_size=B,
                    fixed_base=args.fixed_base,
                    r_min=args.r_min,
                    r_max=args.r_max,
                    lam_scale=args.data_scale,
                    r_schedule=args.r_schedule,
                    time_schedule=args.time_schedule,
                    **schedule_kwargs
                )
                
                # Get one batch and reformat to legacy format
                batch = next(iter(temp_dataloader))
                x0 = batch['x0']
                x1 = batch['x1'] 
                x_t = batch['x_t']
                t = batch['t'].unsqueeze(-1)
                z = batch['z']
                r = batch['r']
                
                return x0, x1, x_t, t, z, r
            
            return mock_sampler
        
        legacy_sampler = create_mock_legacy_sampler(args.bridge, args.dataset)
        
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

        # Plot bridge dynamics
        fig = plot_bridge_dynamics(legacy_sampler, title=f"Bridge Dynamics ({args.bridge}, {args.r_schedule}, {args.time_schedule})")
        plt.savefig(f"{plot_dirs['bridges']}/dynamics_{config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot reverse trajectory
        fig = plot_reverse_trajectory(x1, z_gen, model, mode=sample_bridge_mode)
        plt.savefig(f"{plot_dirs['sampling']}/trajectories_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot generation analysis
        fig = plot_generation_analysis(x0_samples, lam0_batch, lam1_batch, 
                                     title=f"Generation Analysis ({args.arch} + {sample_bridge_mode}, {sample_r_schedule}, {sample_time_schedule})")
        plt.savefig(f"{plot_dirs['analysis']}/generation_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Debug model predictions
        fig = debug_model_predictions(model, legacy_sampler, device=device)
        plt.savefig(f"{plot_dirs['analysis']}/predictions_{config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to organized directories:")
        print(f"  Schedules: {plot_dirs['schedules']}/")
        print(f"  Training:  {plot_dirs['training']}/")
        print(f"  Bridges:   {plot_dirs['bridges']}/")
        print(f"  Sampling:  {plot_dirs['sampling']}/")
        print(f"  Analysis:  {plot_dirs['analysis']}/")


if __name__ == "__main__":
    main() 