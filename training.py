"""
Training Functions for Count-based Flow Matching

Provides training loop and utilities for count-based flow matching models.
"""

import torch
import os
from pathlib import Path
from .datasets import create_dataloader, InfiniteDataLoader


def save_model_checkpoint(model, optimizer, step, losses, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'losses': losses
    }, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")


def train_model(
    model, dataloader, 
    num_iterations=10000, print_interval=2000,
    lr=2e-3, device="cuda", save_every=None, checkpoint_dir=None
):
    """
    Unified training function for count-based flow matching models using DataLoaders
    
    Args:
        model: Neural network model (any BaseCountModel subclass)
        dataloader: PyTorch DataLoader providing training batches
        num_iterations: Number of training iterations
        lr: Learning rate
        device: Device for computation
        save_every: Save model every N steps (None to disable saving)
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        model: Trained model
        losses: List of training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup checkpoint directory if saving is enabled
    if save_every is not None and checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Make dataloader infinite for training
    infinite_loader = InfiniteDataLoader(dataloader)
    
    losses = []
    interval_losses = []
    
    for step in range(num_iterations):
        # Get batch from dataloader
        batch = next(infinite_loader)
        
        # Move to device
        x0 = batch['x0'].to(device)
        x1 = batch['x1'].to(device)
        x_t = batch['x_t'].to(device) 
        M_t = batch['M'].to(device)
        z = batch['z'].to(device)
        t = batch['t'].to(device).unsqueeze(-1)  # Add dimension for broadcasting

        # Training step
        optimizer.zero_grad()
        loss, x0_preds = model.loss(x0, x_t, M_t, z, t)
        loss.backward()
        optimizer.step()


        if step % 100 == 0:
            print(f"t: {t.min()}, {t.max()}, M_t: {M_t.min()}, {M_t.max()}, x_0: {torch.quantile(x0.float(), 0.25)}, {torch.quantile(x0.float(), 0.75)}, x0_preds: {torch.quantile(x0_preds.float(), 0.25)}, {torch.quantile(x0_preds.float(), 0.75)}")
        
        current_loss = loss.item()
        losses.append(current_loss)
        interval_losses.append(current_loss)
        
        # Save checkpoint if enabled
        if save_every is not None and checkpoint_dir is not None and (step + 1) % save_every == 0:
            save_model_checkpoint(model, optimizer, step + 1, losses, Path(checkpoint_dir))
        
        # Print average loss over interval
        if step % print_interval == 0 and step > 0:
            avg_loss = sum(interval_losses) / len(interval_losses)
            print(f"  Step {step:5d}: avg loss = {avg_loss:.4f} (last {len(interval_losses)} steps)")
            interval_losses = []
        elif step == 0:
            print(f"  Step {step:5d}: loss = {current_loss:.4f}")
    
    # Save final checkpoint if enabled
    if save_every is not None and checkpoint_dir is not None:
        save_model_checkpoint(model, optimizer, num_iterations, losses, Path(checkpoint_dir))
    
    # Print final average if there are remaining losses
    if interval_losses:
        avg_loss = sum(interval_losses) / len(interval_losses)
        print(f"  Final:     avg loss = {avg_loss:.4f} (last {len(interval_losses)} steps)")
    else:
        print(f"  Final:     loss = {losses[-1]:.4f}")
    
    return model, losses


def create_training_dataloader(bridge_type, dataset_type, batch_size, d, n_steps, 
                             dataset_size=10000, **kwargs):
    """
    Convenience function to create a training DataLoader
    
    Args:
        bridge_type: "poisson" or "nb"
        dataset_type: "constant_poisson", "random_poisson", or "complex_nb"
        batch_size: Batch size
        d: Count vector dimensionality
        n_steps: Number of diffusion steps
        dataset_size: Size of dataset per epoch
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        DataLoader for training, Dataset instance
    """
    return create_dataloader(
        bridge_type=bridge_type,
        dataset_type=dataset_type,
        batch_size=batch_size,
        size=dataset_size,
        d=d,
        n_steps=n_steps,
        **kwargs
    ) 