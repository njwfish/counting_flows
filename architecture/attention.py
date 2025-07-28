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
        
        # Calculate dimensions for broadcasting inputs
        # We need to know which inputs have shape (batch, output_dim) to split them
        # For now, assume first input dimension matches output_dim for splitting
        self.split_dim = self.output_dim if output_dim is None else output_dim
        
        # Calculate input dimension per token after concatenation
        # Each token gets: projected split input + all broadcasted inputs
        split_input_dim = hidden_dim  # We'll project split inputs to hidden_dim
        broadcast_input_dim = sum(self.in_dims) - self.split_dim  # Remaining inputs get broadcasted
        self.token_input_dim = split_input_dim + broadcast_input_dim
        
        # Projection for split inputs (from raw value to hidden_dim)
        self.split_projection = nn.Linear(1, hidden_dim)
        
        # Position embeddings for each dimension
        self.pos_embeddings = nn.Parameter(torch.randn(self.output_dim, hidden_dim))
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(self.token_input_dim, hidden_dim)
        
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
        # Concatenate all inputs first
        all_inputs = concat_inputs(**kwargs)  # (batch_size, total_input_dim)
        batch_size = all_inputs.shape[0]
        
        # Split inputs into "splittable" and "broadcastable"
        split_inputs = all_inputs[:, :self.split_dim]  # (batch_size, split_dim)
        broadcast_inputs = all_inputs[:, self.split_dim:]  # (batch_size, broadcast_dim)
        
        # Vectorized token creation - much more efficient!
        # Reshape split inputs for vectorized projection
        split_reshaped = split_inputs.unsqueeze(-1)  # (batch_size, output_dim, 1)
        
        # Apply projection to all dimensions at once
        projected_dims = self.split_projection(split_reshaped)  # (batch_size, output_dim, hidden_dim)
        
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