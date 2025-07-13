"""
Clean Training Functions for Count-based Flow Matching with GPU Bridges

Simplified training loop that works with CuPy bridges and normal epochs.
"""

import torch
import cupy as cp
from torch.utils.data import DataLoader
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from bridges.cupy.utils import dlpack_backend

def save_model_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    losses: list, 
    checkpoint_dir: Path
) -> None:
    """Save model checkpoint"""
    checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")


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
        checkpoint_dir: Optional[str] = None,
        config_hash: Optional[str] = None,
        original_cwd: Optional[str] = None
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
            save_every: Save checkpoint every N epochs
            checkpoint_dir: Directory for checkpoints
            config_hash: Config hash for checkpoint naming
            original_cwd: Original working directory for checkpoint saving
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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.config_hash = config_hash
        self.original_cwd = Path(original_cwd) if original_cwd else Path.cwd()
        
        # Setup checkpoint directory
        if self.save_every is not None and self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.losses = []
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
    
    def _save_config_checkpoint(self, model: torch.nn.Module, epoch: int) -> None:
        """Save checkpoint using config hash naming"""
        if self.config_hash is None:
            return
        
        output_dir = self.original_cwd / "outputs" / self.config_hash
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "model.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'epoch': epoch,
            'config_hash': self.config_hash
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Config checkpoint saved: {checkpoint_path}")
    
    def training_step(self, model: torch.nn.Module, bridge: Any, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step"""
        # Extract data from batch
        x_0 = batch['x_0']  # Source counts
        x_1 = batch['x_1']  # Target counts  
        z = None  # No conditioning for simple case
        
        # Convert to CuPy for bridge processing
        x_0_cp, x_1_cp = cp.array(x_0), cp.array(x_1)
        
        # Apply bridge to get diffusion samples
        x_t_cp, M_t_cp, t_cp = bridge(x_0_cp, x_1_cp)
        
        # Convert back to PyTorch tensors for model
        x_0, x_1, x_t, M_t, t = dlpack_backend(x_0_cp, x_1_cp, x_t_cp, M_t_cp, t_cp)
        
        # Training step
        self.optimizer.zero_grad()
        loss, x0_preds = model.loss(x_0, x_t, M_t, t, z)
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
            num_epochs: Number of training epochs
        
        Returns:
            Trained model and list of losses
        """
        # Setup model and optimizer
        model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create dataloader from dataset
        dataloader = self._create_dataloader(dataset)
        
        logging.info(f"Starting training for {self.num_epochs} epochs...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"Dataset size: {len(dataset)} samples")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Batches per epoch: {len(dataloader)}")
        if hasattr(dataset, 'k'):
            logging.info(f"Mixture components: {dataset.k}")
            logging.info(f"Lambda scale: {dataset.lambda_scale}")
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = self.train_epoch(model, bridge, dataloader)
            
            # Print progress
            if (epoch + 1) % self.print_every == 0:
                logging.info(f"Epoch {epoch + 1:3d}/{self.num_epochs}: avg loss = {epoch_loss:.4f}")
            
            # Save checkpoint if enabled and config_hash is provided
            if (self.save_every is not None and 
                self.config_hash is not None and 
                (epoch + 1) % self.save_every == 0):
                self._save_config_checkpoint(model, epoch + 1)
        
        # Save final checkpoint with config_hash if available
        if self.config_hash is not None:
            self._save_config_checkpoint(model, self.num_epochs)
        elif self.save_every is not None and self.checkpoint_dir is not None:
            # Fallback to old checkpoint system
            save_model_checkpoint(
                model, self.optimizer, self.num_epochs, 
                self.losses, self.checkpoint_dir
            )
        
        logging.info("Training completed!")
        return model, self.losses 