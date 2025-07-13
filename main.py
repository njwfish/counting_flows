"""
Clean Hydra-based Main Script for Count-based Flow Matching

Uses Hydra configuration system to coordinate training with GPU bridges and epochs.
"""

import torch
import cupy as cp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import numpy as np
import os
import hashlib
import json

# Capture original working directory before Hydra changes it
ORIGINAL_CWD = Path.cwd().resolve()

from visualization import plot_loss_curve, plot_bridge_marginals, plot_model_samples, save_plots


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def setup_device(device_config: str) -> str:
    """Setup compute device"""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    logging.info(f"Using device: {device}")
    
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    return device


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_config_hash(cfg: DictConfig) -> str:
    """Create a hash from config excluding training parameters"""
    # Create a copy and remove training-specific parameters
    config_copy = OmegaConf.to_container(cfg, resolve=True)
    
    # Remove training parameters that shouldn't affect model identity
    training_params = [
        'num_epochs', 'learning_rate', 'weight_decay', 'batch_size', 
        'print_every', 'save_every', 'checkpoint_dir', 'create_plots',
        'save_model', 'grad_clip_enabled', 'grad_clip_max_norm',
        'shuffle', 'num_workers'
    ]
    
    # Remove training params from trainer config
    if 'trainer' in config_copy:
        for param in training_params:
            config_copy['trainer'].pop(param, None)
    
    # Remove top-level training params
    for param in training_params:
        config_copy.pop(param, None)
    
    # Create hash from remaining config
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]  # Use first 12 chars


def get_output_dir(config_hash: str, original_cwd: Path = None) -> Path:
    """Get output directory for a given config hash"""
    # Use original working directory to ensure outputs are saved outside hydra outputs dir
    if original_cwd is None:
        original_cwd = Path.cwd()
    
    output_dir = original_cwd / "outputs" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_checkpoint(config_hash: str, original_cwd: Path) -> dict:
    """Load checkpoint if it exists"""
    output_dir = get_output_dir(config_hash, original_cwd)
    checkpoint_path = output_dir / "model.pt"
    
    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    return None


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   losses: list, cfg: DictConfig, config_hash: str, original_cwd: Path) -> None:
    """Save checkpoint with config hash"""
    output_dir = get_output_dir(config_hash, original_cwd)
    checkpoint_path = output_dir / "model.pt"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'config_hash': config_hash
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to: {checkpoint_path}")
    


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logging
    setup_logging(cfg.logging.level)
    
    # Set random seeds
    set_random_seeds(cfg.seed)
    
    # Setup device
    device = setup_device(cfg.device)
    
    # Log experiment details
    logging.info("=" * 60)
    logging.info("Count-based Flow Matching with GPU Bridges")
    logging.info("=" * 60)
    logging.info(f"Experiment: {cfg.experiment.name}")
    logging.info(f"Description: {cfg.experiment.description}")
    logging.info(f"Data dimension: {cfg.data_dim}")
    logging.info(f"Diffusion steps: {cfg.n_steps}")
    logging.info(f"Device: {device}")
    
    # Create config hash for checkpointing
    config_hash = create_config_hash(cfg)
    output_dir = get_output_dir(config_hash, ORIGINAL_CWD)
    logging.info(f"Config hash: {config_hash}")
    logging.info(f"Output directory: {output_dir}")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(config_hash, ORIGINAL_CWD)
    if checkpoint:
        logging.info("Found existing checkpoint! Loading model...")
        
        # Load model and return early if training appears complete
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Simple heuristic: if we have losses for expected number of steps, consider complete
        expected_steps = cfg.training.num_epochs * (50000 // cfg.training.batch_size)  # Rough estimate
        if len(checkpoint['losses']) >= expected_steps * 0.9:  # 90% of expected steps
            logging.info("Model appears to be fully trained. Skipping training.")
            trained_model = model
            losses = checkpoint['losses']
            
            # Still create visualizations and save final results
            final_model_path = Path("final_model.pt")
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'config': OmegaConf.to_container(cfg, resolve=True),
                'losses': losses,
                'final_loss': losses[-1] if losses else float('inf')
            }, final_model_path)
            logging.info(f"Final model saved to: {final_model_path}")
            
            # Skip to visualization
            skip_training = True
        else:
            logging.info("Resuming training from checkpoint...")
            skip_training = False
    else:
        logging.info("No existing checkpoint found. Starting fresh training.")
        skip_training = False
    
    # Log configurations
    logging.info("\nConfiguration Details:")
    logging.info(f"Bridge: {cfg.bridge._target_}")
    logging.info(f"Model: {cfg.model._target_}")
    logging.info(f"Dataset: {cfg.dataset._target_}")
    logging.info(f"Trainer: {cfg.training._target_}")
    logging.info(f"Training epochs: {cfg.training.num_epochs}")
    
    # Only proceed with training if not skipping
    if not skip_training:
        # Instantiate bridge using Hydra
        logging.info("\nInstantiating bridge...")
        bridge = hydra.utils.instantiate(cfg.bridge)
        logging.info(f"Bridge instantiated: {type(bridge).__name__}")
        
        # Instantiate model using Hydra
        logging.info("\nInstantiating model...")
        if checkpoint:
            model = hydra.utils.instantiate(cfg.model)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Model loaded from checkpoint")
        else:
            model = hydra.utils.instantiate(cfg.model)
            logging.info(f"Model instantiated: {type(model).__name__}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Instantiate dataset using Hydra
        logging.info("\nInstantiating dataset...")
        dataset = hydra.utils.instantiate(cfg.dataset)
        logging.info(f"Dataset instantiated: {type(dataset).__name__}")
        logging.info(f"Dataset size: {len(dataset)} samples")
        logging.info(f"Data dimension: {dataset.d}")
        if hasattr(dataset, 'k'):
            logging.info(f"Mixture components: {dataset.k}")
            logging.info(f"Lambda scale: {dataset.lambda_scale}")
        
        # Instantiate trainer using Hydra
        logging.info("\nInstantiating trainer...")
        trainer = hydra.utils.instantiate(cfg.training, config_hash=config_hash, original_cwd=str(ORIGINAL_CWD))
        logging.info(f"Trainer instantiated: {type(trainer).__name__}")
        logging.info(f"Batch size: {trainer.batch_size}")
        
        # Load optimizer state if resuming
        if checkpoint:
            # Need to initialize model on device first
            device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Create optimizer and load state
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.optimizer = optimizer
            trainer.losses = checkpoint['losses']
            logging.info("Optimizer state loaded from checkpoint")
        
        # Start training
        logging.info("\nStarting training...")
        trained_model, losses = trainer.train(
            model=model,
            bridge=bridge,
            dataset=dataset
        )
        
        # Save checkpoint after training
        if hasattr(trainer, 'optimizer'):
            save_checkpoint(trained_model, trainer.optimizer, losses, cfg, config_hash, ORIGINAL_CWD)
    
    # Save final results (if not already saved)
    if not skip_training:
        logging.info("\nSaving final results...")
        
        # Save final model
        final_model_path = Path("final_model.pt")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True),
            'losses': losses,
            'final_loss': losses[-1] if losses else float('inf')
        }, final_model_path)
        logging.info(f"Final model saved to: {final_model_path}")
        
        # Save losses
        losses_path = Path("training_losses.pt")
        torch.save(losses, losses_path)
        logging.info(f"Training losses saved to: {losses_path}")
    
    # Create visualizations if enabled
    if cfg.get('create_plots', True):
        logging.info("Creating visualization plots...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plots = {}
        
        # Plot training loss
        if losses:
            loss_fig = plot_loss_curve(losses, title="Training Loss")
            plots['training_loss'] = loss_fig
        
        # If training was skipped, we need to instantiate dataset and bridge
        if skip_training:
            dataset = hydra.utils.instantiate(cfg.dataset)
            bridge = hydra.utils.instantiate(cfg.bridge)
            logging.info("Instantiated dataset and bridge for visualization")
        
        # Get sample data from dataset
        batch_size = 100
        x0_batch = []
        x1_batch = []
        
        for i in range(batch_size):
            sample = dataset[i % len(dataset)]
            x0_batch.append(sample['x_0'])
            x1_batch.append(sample['x_1'])
        
        x0_batch = torch.stack(x0_batch)
        x1_batch = torch.stack(x1_batch)
        
        bridge_fig = plot_bridge_marginals(
            x0_batch=x0_batch,
            x1_batch=x1_batch,
            bridge=bridge,
            title="Bridge Marginals"
        )
        plots['bridge_marginals'] = bridge_fig
        
        # Generate and visualize samples from the trained model
        if not skip_training or checkpoint:
            logging.info("Generating samples from trained model...")
            sample_fig = plot_model_samples(
                model=trained_model,
                bridge=bridge, 
                dataset=dataset,
                n_samples=200,
                title="Generated Samples"
            )
            plots['generated_samples'] = sample_fig

    # Save all plots
    if plots:
        save_plots(plots, str(plots_dir))
        logging.info(f"Plots saved to: {plots_dir}")
    
    # Log final statistics
    if losses:
        final_loss = losses[-1]
        avg_last_100 = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
        logging.info(f"\nTraining completed!")
        logging.info(f"Final loss: {final_loss:.4f}")
        logging.info(f"Average loss (last 100 steps): {avg_last_100:.4f}")
        logging.info(f"Total training steps: {len(losses)}")
    
    logging.info("=" * 60)


def load_and_sample(
    model_path: str,
    bridge_config: DictConfig,
    n_samples: int = 1000,
    device: str = "cuda"
) -> tuple:
    """
    Load trained model and generate samples
    
    Args:
        model_path: Path to saved model
        bridge_config: Bridge configuration for sampling
        n_samples: Number of samples to generate
        device: Device for computation
    
    Returns:
        Generated samples and model
    """
    # Load model and config
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate model
    model_config = DictConfig(config['model'])
    model = hydra.utils.instantiate(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Recreate bridge for sampling
    bridge = hydra.utils.instantiate(bridge_config)
    
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"Model final training loss: {checkpoint.get('final_loss', 'unknown')}")
    
    # Generate samples (this would need to be implemented based on your sampling needs)
    logging.info(f"Generating {n_samples} samples...")
    # TODO: Implement sampling logic based on your specific requirements
    
    return None, model  # Placeholder return


if __name__ == "__main__":
    main() 