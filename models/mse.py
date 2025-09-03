import torch
import torch.nn as nn

class MSELoss(nn.Module):
    """
    Simple MSE loss that works with arbitrary architectures and clean input interface.
    """
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture

    def forward(self, inputs):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            
        Returns:
            Prediction tensor
        """
        # MSE is deterministic - just pass inputs directly (no noise)
        return self.architecture(**inputs)

    def sample(self, **kwargs):
        """Deterministic sampling (just return the forward pass)."""
        return self.forward(kwargs)

    def loss(self, target, inputs):
        """Compute MSE loss between prediction and target."""
        # MSE is deterministic - just pass inputs directly (no noise)
        prediction = self.architecture(**inputs)
        return nn.functional.mse_loss(prediction, target.float())