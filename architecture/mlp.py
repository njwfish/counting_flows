import torch
import torch.nn as nn
from .utils import concat_inputs
from typing import Iterable

class MLP(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, layers=2):
        super().__init__()
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        
        self.in_dims = in_dims
        total_input_dim = sum(in_dims)

        print(f"in_dims: {in_dims}")
        print(f"total_input_dim: {total_input_dim}")
        
        
        # Support list outputs by computing total output dimension
        if isinstance(out_dim, Iterable):
            self.out_shape = out_dim
            total_output_dim = 1
            for dim in out_dim:
                total_output_dim *= dim
        else:
            self.out_shape = None
            total_output_dim = out_dim

        print(f"out_dim: {out_dim}")
        print(f"total_output_dim: {total_output_dim}")
        
        self.layers = nn.ModuleList([
            nn.Linear(total_input_dim, hidden_dim),
            nn.SELU()
        ])
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SELU())
        self.layers.append(nn.Linear(hidden_dim, total_output_dim))

    def forward(self, **kwargs): 
        """
        Forward pass with flexible keyword arguments.
        
        Args:
            **kwargs: Named tensors that will be concatenated
            
        Returns:
            Output tensor, reshaped if out_dim was a list
        """
        # Concatenate all inputs using utility function
        x = concat_inputs(**kwargs)
        
        for layer in self.layers:
            x = layer(x)

        # Reshape output if out_dim was specified as a list
        if self.out_shape is not None:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, *self.out_shape)

        return x 