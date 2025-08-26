"""
Clean Training Functions for Flow-based Models with Modular Architecture

Simplified training loop that works with various bridges (CFM, counting flows, etc.) and loss functions.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


class FlowTrainer:
    """
    Clean, modular trainer for flow-based models.
    Works with any bridge (CFM, counting flows, etc.) and loss function (energy score, MSE, etc.).
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        num_epochs: int = 100,
        device: str = "cuda",
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
        print_every: int = 10,
        save_every: Optional[int] = None,
        output_dir: Optional[str] = None
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.print_every = print_every
        self.save_every = save_every
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Training state
        self.losses = []
        self.start_epoch = 0
        self.optimizer = None
    
    def _create_dataloader(self, dataset, shuffle=None):
        """Create DataLoader from dataset"""
        if shuffle is None:
            shuffle = self.shuffle
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def _save_checkpoint(self, model: torch.nn.Module, current_epoch: int) -> None:
        """Save training checkpoint"""
        if self.output_dir is None:
            return
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'current_epoch': current_epoch,
            'total_epochs': self.num_epochs
        }
        
        checkpoint_path = self.output_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
    
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

        
        if z is not None:
            inputs['z'] = z.to(self.device)
        
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
    
    def train(
        self,
        model: torch.nn.Module,
        bridge: Any,
        dataset: Any,
    ) -> Tuple[torch.nn.Module, list]:
        """
        Main training loop
        
        Args:
            model: Neural network model
            bridge: CuPy bridge for diffusion process
            dataset: PyTorch dataset
        
        Returns:
            Trained model and list of losses
        """
        # Setup model and optimizer
        model = model.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Create dataloader from dataset
        dataloader = self._create_dataloader(dataset)
        
        logging.info(f"Training from epoch {self.start_epoch + 1} to {self.num_epochs}")
        logging.info(f"Device: {self.device}, Batch size: {self.batch_size}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            model.train()
            epoch_loss = self.train_epoch(epoch, model, bridge, dataloader, dataset)
            
            # Print progress
            if (epoch + 1) % self.print_every == 0:
                logging.info(f"Epoch {epoch + 1:3d}/{self.num_epochs}: avg loss = {epoch_loss:.4f}")
            
            # Save checkpoint periodically
            if (self.save_every is not None and 
                (epoch + 1) % self.save_every == 0):
                self._save_checkpoint(model, epoch + 1)
        
        # Save final checkpoint
        self._save_checkpoint(model, self.num_epochs)
        
        logging.info("Training completed!")
        return model, self.losses 