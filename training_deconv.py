"""
Clean Training Functions for Flow-based Models with Modular Architecture

Simplified training loop that works with various bridges (CFM, counting flows, etc.) and loss functions.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from training import Trainer

class DeconvTrainer(Trainer):
    """
    Clean, modular trainer for flow-based models.
    Works with any bridge (CFM, counting flows, etc.) and loss function (energy score, MSE, etc.).
    """
    
    def __init__(
        self,
        **kwargs
    ):
        """
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            device: Device for model computation
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle dataset
            num_workers: Number of DataLoader workers
            print_every: Print loss every N epochs
            save_every: Save checkpoint every N epochs (None to disable)
            output_dir: Output directory for checkpoints
        """
        super().__init__(
            **kwargs
        )
    
    def training_step(self, model: torch.nn.Module, bridge: Any, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step"""
        # Extract data from batch
        x_0 = batch['x_0'].to(self.device)  # Target counts
        x_1 = batch['x_1'].to(self.device)  # Source counts  
        z = batch.get('z', None)  # Conditioning (optional)
        
        # Apply bridge to get diffusion samples
        t, x_t, target = bridge(x_0, x_1)
        
        inputs = {
            "t": t,
            "x_t": x_t,
        }
        target = batch['X_0'] - x_0.sum(dim=1) if bridge.delta else batch['X_0']

        for key in batch:
            if key != 'x_0' and key != 'x_1':
                inputs[key] = batch[key].squeeze(0).to(self.device)
                if 'key' == 'class_emb' and self.classifier_free_guidance_prob > 0:
                    # random mask out prob of the class embeddings
                    mask = torch.rand(inputs[key].shape[0]) < self.classifier_free_guidance_prob
                    inputs[key][mask] = 0
        
        # Training step
        self.optimizer.zero_grad()
        loss = model.loss(target, inputs)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, epoch: int, model: torch.nn.Module, bridge: Any, dataloader: DataLoader, dataset: Any) -> float:
        """Train for one epoch"""
        epoch_losses = []

        # e step
        model.eval()
        batches = []
        for batch in dataloader:
            x_1 = batch['x_1'].to(self.device).reshape(-1, batch['x_1'].shape[-1])
            z = batch.get('z', None)
            if z is not None:
                z = z.to(self.device).reshape(-1, z.shape[-1])

            batch['X_0'] = batch['X_0'].to(self.device)
            context = {'z': z, 'target_sum': batch['X_0']}
            batch['x_0'] = bridge.sampler(
                x_1, context, model
            )[0]
            batch['x_1'] = x_1
            batch['z'] = z

            batches.append(batch)

        # m step
        model.train()
        for batch in batches:
            # sample time step
            batch_loss = self.training_step(model, bridge, batch)
            epoch_losses.append(batch_loss)
            self.losses.append(batch_loss)
        
        return sum(epoch_losses) / len(epoch_losses)
    