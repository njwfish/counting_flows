import torch
import torch.nn as nn
import torch.nn.functional as F

from bridges.baselines.dfm import DiscreteFlowBridge

class DFMLoss(nn.Module):
    """
    Discrete Flow model with cross entropy loss.
    Works with arbitrary architectures and clean input interface.
    No embeddings - treats discrete inputs as regular integers.
    """
    
    def __init__(self, architecture, min_value=0, value_range=128, num_steps=10, context_dim=1):
        """
        Args:
            architecture: MLP architecture that outputs (data_dim, vocab_size) shaped logits
        """
        super().__init__()
        self.architecture = architecture
        self.min_value = min_value
        self.value_range = value_range
        self.num_steps = num_steps
        print(f"min_value: {self.min_value}, value_range: {self.value_range}, num_steps: {self.num_steps}")
        self.bridge = DiscreteFlowBridge(n_steps=num_steps)
    
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
        return self.bridge.sampler(kwargs['x_t'], z={'z': kwargs['t']}, model=self)
    
    def loss(self, target, inputs):
        """
        Compute cross entropy loss.
        
        Args:
            target: Target discrete tokens (batch_size, data_dim)
            inputs: Dict with inputs
            
        Returns:
            Cross entropy loss scalar
        """
        x_1 = inputs['x_t']
        x_0 = target
        out = self.bridge(x_0, x_1)
        inner_inputs = out['inputs']
        inner_inputs['z'] = inputs['t']
        logits = self.forward(inner_inputs)

        loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_0.long().flatten(0, 1))
        return loss
