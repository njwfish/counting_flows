import torch
import torch.nn as nn
from .utils import concat_inputs
from typing import Iterable


class AttentionArch(nn.Module):
    """
    BERT-like attention architecture with flexible inputs/outputs.
    
    Uses standard PyTorch TransformerEncoder with intelligent input handling:
    - Inputs shaped (batch_size, output_dim) are split into per-dimension tokens
    - Other inputs are broadcasted/concatenated to each token
    - Learnable position embeddings for each dimension
    - MLP head for final predictions
    """
    
    def __init__(
        self, 
        in_dims, 
        hidden_dim, 
        out_dim, 
        num_heads=4, 
        num_layers=2,
        dropout=0.1,
        output_dim=None  # For determining which inputs to split
    ):
        super().__init__()
        
        # Convert Hydra configs to regular Python types
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        else:
            in_dims = list(in_dims)  # Convert ListConfig to list
            
        if hasattr(out_dim, '__iter__') and not isinstance(out_dim, int):
            # out_dim is a list/ListConfig
            out_dim = list(out_dim)  # Convert ListConfig to list
            self.out_shape = out_dim
            total_output_dim = 1
            for dim in out_dim:
                total_output_dim *= int(dim)  # Convert to int
            self.output_dim = int(out_dim[0])  # First dimension for splitting
        else:
            self.out_shape = None
            total_output_dim = int(out_dim)
            self.output_dim = int(out_dim)
            
        self.in_dims = [int(dim) for dim in in_dims]  # Convert all to int
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Calculate dimensions from input specification
        # Assume inputs matching output_dim are splittable, others are broadcastable
        splittable_dim = sum(dim for dim in self.in_dims if dim == self.output_dim)
        broadcastable_dim = sum(dim for dim in self.in_dims if dim != self.output_dim)
        
        # Calculate split features per dimension
        self.split_features_per_dim = splittable_dim // self.output_dim if splittable_dim > 0 else 0
        
        # Create projection layers during init
        if self.split_features_per_dim > 0:
            self.split_projection = nn.Linear(self.split_features_per_dim, hidden_dim)
        else:
            self.split_projection = None
            
        # Token input dimension and projection
        self.token_input_dim = hidden_dim + broadcastable_dim
        self.input_projection = nn.Linear(self.token_input_dim, hidden_dim)
        
        # Cached input mapping for efficiency (keys only, not dimensions)
        self._input_mapping_cached = False
        self._splittable_keys = []
        self._broadcastable_keys = []
        
        # Position embeddings for each dimension
        self.pos_embeddings = nn.Parameter(torch.randn(self.output_dim, hidden_dim))
        
        # Standard PyTorch transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP head for final predictions
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, total_output_dim // self.output_dim)  # Output per dimension
        )
    
    def forward(self, **kwargs):
        """
        Forward pass with intelligent input handling.
        
        Args:
            **kwargs: Named tensors with flexible shapes
            
        Returns:
            Output tensor, reshaped if out_dim was a list
        """
        batch_size = next(iter(kwargs.values())).shape[0]
        
        # Cache input mapping on first forward pass for efficiency
        if not self._input_mapping_cached:
            for name, tensor in kwargs.items():
                if tensor.shape[-1] == self.output_dim:  # Data-dimension inputs (x_t, M_t, etc.)
                    self._splittable_keys.append(name)
                else:  # Other inputs (t, noise, etc.)
                    self._broadcastable_keys.append(name)
            self._input_mapping_cached = True
        
        # Use cached mapping for efficient splitting
        splittable_tensors = [kwargs[key] for key in self._splittable_keys]
        broadcastable_tensors = [kwargs[key] for key in self._broadcastable_keys]
        
        # Concatenate splittable and broadcastable separately
        split_inputs = torch.cat(splittable_tensors, dim=-1) if splittable_tensors else torch.empty(batch_size, 0)
        broadcast_inputs = torch.cat(broadcastable_tensors, dim=-1) if broadcastable_tensors else torch.empty(batch_size, 0)
        
        # Verify dimensions match expectations (debug check)
        if split_inputs.shape[1] > 0:
            expected_split_features = split_inputs.shape[1] // self.output_dim
            assert expected_split_features == self.split_features_per_dim, \
                f"Split features mismatch: expected {self.split_features_per_dim}, got {expected_split_features}"
        
        # Vectorized token creation - handle multiple splittable inputs
        if split_inputs.shape[1] > 0 and self.split_projection is not None:
            # Reshape to separate per-dimension tokens: (batch_size, split_features, output_dim)  
            split_reshaped = split_inputs.view(batch_size, self.split_features_per_dim, self.output_dim)
            # Transpose to (batch_size, output_dim, split_features) for per-dimension processing
            split_reshaped = split_reshaped.transpose(1, 2)  # (batch_size, output_dim, split_features)
            
            # Apply projection to all dimensions at once
            projected_dims = self.split_projection(split_reshaped)  # (batch_size, output_dim, hidden_dim)
        else:
            # No splittable inputs - create zero projections
            projected_dims = torch.zeros(batch_size, self.output_dim, self.hidden_dim, device=next(self.parameters()).device)
        
        # Broadcast other inputs to all tokens
        if broadcast_inputs.shape[1] > 0:
            # Expand broadcast inputs to match token dimensions
            broadcast_expanded = broadcast_inputs.unsqueeze(1).expand(
                batch_size, self.output_dim, -1
            )  # (batch_size, output_dim, broadcast_dim)
            
            # Concatenate projected dims with broadcasted inputs
            token_sequence = torch.cat([projected_dims, broadcast_expanded], dim=-1)
        else:
            token_sequence = projected_dims
        
        # Project to transformer dimension
        token_sequence = self.input_projection(token_sequence)  # (batch_size, output_dim, hidden_dim)
        
        # Add position embeddings
        token_sequence = token_sequence + self.pos_embeddings.unsqueeze(0)
        
        # Apply transformer
        attention_output = self.transformer(token_sequence)  # (batch_size, output_dim, hidden_dim)
        
        # Apply head to each token
        output = self.head(attention_output)  # (batch_size, output_dim, output_per_dim)
        
        # Reshape if needed
        if self.out_shape is not None:
            batch_size = output.shape[0]
            output = output.reshape(batch_size, *self.out_shape)
        else:
            # Flatten to (batch_size, output_dim * output_per_dim)
            output = output.reshape(batch_size, -1)
        
        return output 