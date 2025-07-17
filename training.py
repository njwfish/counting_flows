"""
Clean Training Functions for Count-based Flow Matching with GPU Bridges

Simplified training loop that works with CuPy bridges and normal epochs.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


class CountFlowTrainer:
    """
    Clean trainer for count-based flow matching with GPU bridges
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
    
    def _create_dataloader(self, dataset):
        """Create DataLoader from dataset"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
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
        x_0 = batch['x_0'].to(self.device)  # Source counts
        x_1 = batch['x_1'].to(self.device)  # Target counts  
        z = None  # No conditioning for simple case
        
        # Apply bridge to get diffusion samples
        x_t, M_t, t = bridge(x_0, x_1)
        
        # Training step
        self.optimizer.zero_grad()
        loss = model.loss(x_0, x_t, M_t, t, z)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, model: torch.nn.Module, bridge: Any, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        epoch_losses = []
        
        for batch in dataloader:
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
            epoch_loss = self.train_epoch(model, bridge, dataloader)
            
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