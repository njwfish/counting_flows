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
from .models import EnergyScorePosterior
from .training import train_model, create_training_dataloader
from .bridges.skellam import SkellamBridge
from .bridges.reflected import ReflectedPoissonBDBridge
from .bridges.constrained import SkellamMeanConstrainedBridge
from .visualization import (
    plot_time_spacing_comparison,
    plot_generation_analysis, plot_loss_curve,
    plot_full_reverse_trajectories, plot_true_marginal_distributions
)


def load_model_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint if it exists"""
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            step = checkpoint.get('step', 0)
            losses = checkpoint.get('losses', [])
            print(f"Loaded checkpoint from {checkpoint_path} (step {step})")
            model = model.to(device)
            return model, step, losses
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return model, 0, []
    return model, 0, []


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("model_step_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by step number to get the latest
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1]


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
        args.iterations = 10_000
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
    elif args.arch == "iqn":
        model = IQNPosterior(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: Implicit Quantile Networks")
    elif args.arch == "mmd":
        model = EnergyScorePosterior(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: Maximum Mean Discrepancy")
    else:
        model = MLERegressor(x_dim=args.data_dim, context_dim=context_dim, hidden=args.hidden)
        print("Model: MLE regressor")
    
    # Create training DataLoader based on bridge mode
    dataset_size = args.dataset_size if args.dataset_size is not None else max(args.iterations, 10000)
    
    base_mode = "fixed base" if args.fixed_base else "random base"
    print(f"Dataset: {args.dataset.title()} ({args.data_dim}D, {base_mode})")
    
    # Symmetric bridge creation - all bridges get the same base parameters
    if args.bridge == "poisson_bd":
        train_dataloader, train_dataset = create_training_dataloader(
            bridge_type="poisson_bd",
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            d=args.data_dim,
            n_steps=args.steps,
            dataset_size=dataset_size,
            fixed_base=args.fixed_base,
            time_schedule=args.time_schedule,
            lam0=args.lam0,
            lam1=args.lam1,
            schedule_type=args.bd_schedule,
            **schedule_kwargs
        )
        print(f"Bridge: Poisson Birth-Death for training")
        print(f"  Lambda: {args.lam0:.1f} → {args.lam1:.1f}")
        print(f"  Lambda schedule: {args.bd_schedule}")
        print(f"  Time schedule: {args.time_schedule}")
    elif args.bridge == "reflected_bd":
        train_dataloader, train_dataset = create_training_dataloader(
            bridge_type="reflected_bd",
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            d=args.data_dim,
            n_steps=args.steps,
            dataset_size=dataset_size,
            fixed_base=args.fixed_base,
            time_schedule=args.time_schedule,
            lam0=args.lam0,
            lam1=args.lam1,
            schedule_type=args.bd_schedule,
            **schedule_kwargs
        )
        print(f"Bridge: Reflected Birth-Death for training")
        print(f"  Lambda: {args.lam0:.1f} → {args.lam1:.1f}")
        print(f"  Lambda schedule: {args.bd_schedule}")
        print(f"  Time schedule: {args.time_schedule}")
    elif args.bridge == "poisson_bd_mean":
        train_dataloader, train_dataset = create_training_dataloader(
            bridge_type="poisson_bd_mean",
            dataset_type=args.dataset,
            batch_size=args.batch_size,
            d=args.data_dim,
            n_steps=args.steps,
            dataset_size=dataset_size,
            fixed_base=args.fixed_base,
            time_schedule=args.time_schedule,
            lam0=args.lam0,
            lam1=args.lam1,
            schedule_type=args.bd_schedule,
            mh_sweeps=args.mh_sweeps,
            **schedule_kwargs
        )
        print(f"Bridge: Mean-Constrained Poisson Birth-Death for training")
        print(f"  Lambda: {args.lam0:.1f} → {args.lam1:.1f}")
        print(f"  Lambda schedule: {args.bd_schedule}")
        print(f"  Time schedule: {args.time_schedule}")
    else:
        raise ValueError(f"Unknown bridge type: {args.bridge}. Supported types: poisson_bd, reflected_bd, poisson_bd_mean")
    
    # Setup checkpoints and model loading
    checkpoint_dir = "checkpoints"
    config_str = f"{args.arch}_{args.bridge}_{args.r_schedule}_{args.time_schedule}"
    model_checkpoint_dir = f"{checkpoint_dir}/{config_str}"
    
    start_step = 0
    initial_losses = []
    
    # Try to load model if --load flag is set
    if args.load:
        latest_checkpoint = find_latest_checkpoint(model_checkpoint_dir)
        if latest_checkpoint:
            model, start_step, initial_losses = load_model_checkpoint(model, latest_checkpoint, device)
            print(f"Resuming training from step {start_step}")
        else:
            print(f"No checkpoint found in {model_checkpoint_dir}, starting from scratch")
    
    # Configure checkpoint saving (save every 10,000 steps)
    save_every = max(args.iterations // 10, 1000)  # Save at least every 1000 steps, or 10 times during training
    
    # Train
    print(f"\nTraining model...")
    if start_step < args.iterations:
        remaining_iterations = args.iterations - start_step
        print(f"Training for {remaining_iterations} more iterations (total: {args.iterations})")
        model, new_losses = train_model(
            model, train_dataloader,
            num_iterations=remaining_iterations,
            lr=args.lr, 
            device=device,
            save_every=save_every,
            checkpoint_dir=model_checkpoint_dir
        )
        losses = initial_losses + new_losses
    else:
        print(f"Model already trained for {start_step} steps, skipping training")
        losses = initial_losses
    
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
            batch_size=args.gen_samples * 10,  # Generate one large batch
            homogeneous=True,  # Different parameters per sample
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
        
        # Create bridge instance for sampling (same as training)
        bridge_kwargs = {
            'n_steps': args.steps,
            'lam0': args.lam0,
            'lam1': args.lam1,
            'schedule_type': args.bd_schedule,
            'time_schedule': sample_time_schedule,
            **schedule_kwargs
        }
        
        if sample_bridge_mode == "poisson_bd":
            bridge = SkellamBridge(**bridge_kwargs)
        elif sample_bridge_mode == "reflected_bd":
            bridge = ReflectedPoissonBDBridge(**bridge_kwargs)
        elif sample_bridge_mode == "poisson_bd_mean":
            bridge_kwargs['mh_sweeps'] = args.mh_sweeps
            bridge = SkellamMeanConstrainedBridge(**bridge_kwargs)
        else:
            raise ValueError(f"Unknown sampling bridge: {sample_bridge_mode}")
        
        # All bridges can optionally receive mu0 for consistency
        mu0 = x0_target.float().mean(0)  # (d,)
        
        # Use bridge's built-in reverse sampler
        if sample_bridge_mode == "poisson_bd_mean":
            x0_samples, trajectory, x_hat_trajectory, M_trajectory = bridge.reverse_sampler(
                x1=x1_batch,
                z=z_batch,
                mu0=mu0,
                model=model,
                device=device,
                return_trajectory=True,
                return_x_hat=True,
                return_M=True,
            )
        else:
            # Regular and reflected bridges - mu0 is optional but we can pass it
            x0_samples, trajectory, x_hat_trajectory, M_trajectory = bridge.reverse_sampler(
                x1=x1_batch,
                z=z_batch,
                model=model,
                device=device,
                return_trajectory=True,
                return_x_hat=True,
                return_M=True,
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
        
        fig = plot_time_spacing_comparison(K=args.steps, title="Time Spacing Comparison") 
        plt.savefig(f"{plot_dirs['schedules']}/time_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot losses
        fig = plot_loss_curve(losses, f"Training Loss ({args.arch} + {args.bridge} bridge)")
        plt.savefig(f"{plot_dirs['training']}/loss_{config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot reverse trajectory using full trajectory data
        fig = plot_full_reverse_trajectories(
            trajectory=trajectory,
            x_hat_trajectory=x_hat_trajectory,
            M_trajectory=M_trajectory,
            x0_target=x0_target,
            x1_batch=x1_batch,
            mode=sample_bridge_mode,
            K=args.steps,
            title=f"Full Reverse Trajectories ({sample_bridge_mode} bridge)"
            
        )
        plt.savefig(f"{plot_dirs['sampling']}/trajectories_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot generation analysis
        fig = plot_generation_analysis(x0_samples, x0_target, x1_batch, 
                                     title=f"Generation Analysis ({args.arch} + {sample_bridge_mode}, {sample_r_schedule}, {sample_time_schedule})")
        plt.savefig(f"{plot_dirs['analysis']}/generation_{sample_config_str}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot true marginal distributions from bridge
        bridge_collate = None
        if args.bridge == "poisson_bd":
            bridge_collate = SkellamBridge(
                n_steps=args.steps,
                lam0=args.lam0,
                lam1=args.lam1,
                schedule_type=args.bd_schedule,
                time_schedule=args.time_schedule,
                homogeneous_time=False,  # We want different times
                **schedule_kwargs
            )
        elif args.bridge == "reflected_bd":
            bridge_collate = ReflectedPoissonBDBridge(
                n_steps=args.steps,
                lam0=args.lam0,
                lam1=args.lam1,
                schedule_type=args.bd_schedule,
                time_schedule=args.time_schedule,
                homogeneous_time=False,  # We want different times
                **schedule_kwargs
            )
        elif args.bridge == "poisson_bd_mean":
            bridge_collate = SkellamMeanConstrainedBridge(
                n_steps=args.steps,
                lam0=args.lam0,
                lam1=args.lam1,
                schedule_type=args.bd_schedule,
                time_schedule=args.time_schedule,
                homogeneous_time=False,  # We want different times
                mh_sweeps=args.mh_sweeps,
                **schedule_kwargs
            )
        
        if bridge_collate is not None:
            try:
                fig = plot_true_marginal_distributions(
                    x0_batch=x0_target[:100],  # Use subset for speed
                    x1_batch=x1_batch[:100],
                    z_batch=z_batch[:100],
                    bridge_collate=bridge_collate,
                    times=[0.0, 0.25, 0.5, 0.75, 1.0],
                    n_samples=500,
                    title=f"True Bridge Marginals ({args.bridge})"
                )
                plt.savefig(f"{plot_dirs['analysis']}/bridge_marginals_{config_str}.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Bridge marginal distributions plotted.")
            except Exception as e:
                print(f"Could not plot bridge marginals: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Bridge marginal plots not supported for bridge type: {args.bridge}")
        
        print(f"Plots saved to organized directories:")
        print(f"  Schedules: {plot_dirs['schedules']}/")
        print(f"  Training:  {plot_dirs['training']}/")
        print(f"  Sampling:  {plot_dirs['sampling']}/")
        print(f"  Analysis:  {plot_dirs['analysis']}/")
        print(f"\nNote: Some debug plots were removed due to deprecated code dependencies.")


if __name__ == "__main__":
    main() 