"""
Discrete Flow Model with Cross Entropy Loss

Simple discrete flow model that works exactly like MSE - no embeddings,
just passes inputs directly to architecture. Uses cross entropy loss.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Discrete Flow model with cross entropy loss.
    Works with arbitrary architectures and clean input interface.
    No embeddings - treats discrete inputs as regular integers.
    """
    
    def __init__(self, architecture, min_value=0, value_range=128):
        """
        Args:
            architecture: MLP architecture that outputs (data_dim, vocab_size) shaped logits
        """
        super().__init__()
        self.architecture = architecture
        self.min_value = min_value
        self.value_range = value_range
        print(f"min_value: {self.min_value}, value_range: {self.value_range}")
    
    def forward(self, inputs):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            
        Returns:
            Logits tensor of shape (batch_size, data_dim, vocab_size)
        """
        return self.architecture(**inputs)
    
    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        Actually samples from the predicted distribution.
        """
        logits = self.forward(kwargs)  # (batch_size, data_dim, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        samples = torch.distributions.Categorical(probs=probs).sample()
        return samples + self.min_value 
    
    def loss(self, target, inputs):
        """
        Compute cross entropy loss.
        
        Args:
            target: Target discrete tokens (batch_size, data_dim)
            inputs: Dict with inputs
            
        Returns:
            Cross entropy loss scalar
        """
        logits = self.forward(inputs)  # (batch_size, data_dim, vocab_size)
        target = target - self.min_value
        target = torch.clamp(target, min=0, max=self.value_range)
        
        # Flatten for cross entropy: (batch_size * data_dim, vocab_size) and (batch_size * data_dim,)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = target.reshape(-1).long()  # Convert to long for cross entropy
        
        loss = nn.functional.cross_entropy(logits_flat, target_flat)
        return loss 