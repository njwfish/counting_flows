"""
Clean Hydra-based Main Script for Count-based Flow Matching

Simple orchestration of training, evaluation, and visualization.
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import numpy as np

# Capture original working directory before Hydra changes it
ORIGINAL_CWD = Path.cwd().resolve()

from visualization import plot_loss_curve, plot_bridge_marginals, plot_model_samples, save_plots
from eval import generate_evaluation_data, compute_evaluation_metrics, log_evaluation_summary


def setup_environment(cfg: DictConfig) -> str:
    """Setup logging, seeds, and device"""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Setup device
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA requested but not available, falling back to CPU")
    
    logging.info(f"Using device: {device}")
    return device


def get_checkpoint_info(cfg: DictConfig) -> tuple:
    """Get checkpoint path and load if exists"""
    from pathlib import Path
    import hashlib
    import json
    
    # Create config hash (excluding training params)
    config_copy = OmegaConf.to_container(cfg, resolve=True)
    training_params = [
        'num_epochs', 'learning_rate', 'weight_decay', 'batch_size', 
        'print_every', 'save_every', 'checkpoint_dir', 'create_plots',
        'save_model', 'grad_clip_enabled', 'grad_clip_max_norm',
        'shuffle', 'num_workers'
    ]
    
    if 'trainer' in config_copy:
        for param in training_params:
            config_copy['trainer'].pop(param, None)
    for param in training_params:
        config_copy.pop(param, None)
    
    config_str = json.dumps(config_copy, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    # Setup output directory
    output_dir = ORIGINAL_CWD / "outputs" / config_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
    
    # Check for checkpoint
    checkpoint_path = output_dir / "model.pt"
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info(f"Found checkpoint: {checkpoint_path}")
    
    return output_dir, checkpoint


def is_training_complete(checkpoint: dict, total_epochs: int) -> bool:
    """Check if training is complete based on saved epoch info"""
    if not checkpoint:
        return False
    
    current_epoch = checkpoint.get('current_epoch', 0)
    return current_epoch >= total_epochs


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   losses: list, current_epoch: int, cfg: DictConfig, output_dir: Path) -> None:
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'current_epoch': current_epoch,
        'total_epochs': cfg.training.num_epochs,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }
    
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function - simple orchestration"""
    
    # Setup environment
    device = setup_environment(cfg)
    
    # Get checkpoint info
    output_dir, checkpoint = get_checkpoint_info(cfg)
    
    # Check if training is complete
    training_complete = is_training_complete(checkpoint, cfg.training.num_epochs)
    
    # Instantiate everything
    bridge = hydra.utils.instantiate(cfg.bridge)
    dataset = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)
    
    logging.info(f"Experiment: {cfg.experiment.name}")
    logging.info(f"Model: {type(model).__name__} ({sum(p.numel() for p in model.parameters()):,} params)")
    logging.info(f"Dataset: {len(dataset)} samples, {dataset.d}D")
    
    # Training
    if not training_complete:
        logging.info("Starting training...")
        trainer = hydra.utils.instantiate(cfg.training, output_dir=str(output_dir))
        
        # Load from checkpoint if available
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.losses = checkpoint['losses']
            trainer.start_epoch = checkpoint['current_epoch']
            
            # Setup optimizer and load its state
            model = model.to(device)
            trainer.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Resuming from epoch {trainer.start_epoch}")
        
        trained_model, losses = trainer.train(model=model, bridge=bridge, dataset=dataset)
    else:
        logging.info("Training already complete, loading model...")
        model.load_state_dict(checkpoint['model_state_dict'])
        trained_model = model
        losses = checkpoint['losses']
    
    # Save final model
    final_model_path = Path("final_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'losses': losses
    }, final_model_path)
    
    # Evaluation and visualization
    if cfg.get('create_plots', True):
        logging.info("Generating evaluation and plots...")
        
        # Generate evaluation data
        eval_data = generate_evaluation_data(trained_model, bridge, dataset, n_samples=200)
        metrics = compute_evaluation_metrics(eval_data)
        log_evaluation_summary(eval_data, metrics)
        
        # Create plots
        plots = {}
        plots['training_loss'] = plot_loss_curve(losses, title="Training Loss")
        plots['bridge_marginals'] = plot_bridge_marginals(
            x0_batch=eval_data['x0_target'][:100], 
            x1_batch=eval_data['x1_batch'][:100], 
            bridge=bridge, 
            title="Bridge Marginals"
        )
        
        sample_figs = plot_model_samples(eval_data, title="Generated Samples")
        plots['model_trajectories'] = sample_figs['trajectories']
        plots['model_distributions'] = sample_figs['distributions']
        
        # Save plots
        plots_dir = output_dir / "plots"
        save_plots(plots, str(plots_dir))
        logging.info(f"Plots saved to: {plots_dir}")
    
    # Final summary
    final_loss = losses[-1] if losses else float('inf')
    logging.info(f"Training completed! Final loss: {final_loss:.4f}")


if __name__ == "__main__":
    main() 