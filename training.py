"""
Training Functions for Count-based Flow Matching

Provides training loop and utilities for count-based flow matching models.
"""

import torch
from .datasets import create_dataloader, InfiniteDataLoader


def train_model(
    model, dataloader, 
    num_iterations=10000, print_interval=2000,
    lr=2e-3, device="cuda"
):
    """
    Unified training function for count-based flow matching models using DataLoaders
    
    Args:
        model: Neural network model (any BaseCountModel subclass)
        dataloader: PyTorch DataLoader providing training batches
        num_iterations: Number of training iterations
        lr: Learning rate
        device: Device for computation
    
    Returns:
        model: Trained model
        losses: List of training losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        z = batch['z'].to(device)
        t = batch['t'].to(device).unsqueeze(-1)  # Add dimension for broadcasting

        if step == 0:
            print(
                x0.float().mean(0), x1.float().mean(0), "\n",
                x_t.float().mean(0), x0.float().mean(0) * (1 - t.float().mean()) + x1.float().mean(0) * t.float().mean(), "\n",
                t.float().mean(), z.float().mean(0)
            )
        # Training step
        optimizer.zero_grad()
        loss = model.loss(x0, x_t, z, t)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        interval_losses.append(current_loss)
        
        # Print average loss over interval
        if step % print_interval == 0 and step > 0:
            avg_loss = sum(interval_losses) / len(interval_losses)
            print(f"  Step {step:5d}: avg loss = {avg_loss:.4f} (last {len(interval_losses)} steps)")
            interval_losses = []
        elif step == 0:
            print(f"  Step {step:5d}: loss = {current_loss:.4f}")
    
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