import torch
import torch.nn as nn
from .utils import concat_inputs
from typing import Literal


class MeanPooledMLP(nn.Module):
    """Mean-pooled MLP that gives each position access to global context efficiently."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        # Input is concatenated with mean-pooled representation
        self.mlp = nn.ModuleList([nn.Linear(2 * in_dim, hidden_dim), nn.SELU()])
        for _ in range(layers - 1):
            self.mlp.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        self.mlp.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, in_dim)
        Returns:
            (batch_size, seq_len, out_dim)
        """
        # Compute mean-pooled representation
        pooled_rep = x.mean(dim=1)  # (batch_size, in_dim)
        
        # Concatenate pooled representation to each position
        pooled_expanded = pooled_rep.unsqueeze(1).expand(-1, x.shape[1], -1)
        x_with_context = torch.cat([x, pooled_expanded], dim=-1)  # (batch_size, seq_len, 2*in_dim)
        
        # Apply MLP
        for layer in self.mlp:
            x_with_context = layer(x_with_context)
        
        return x_with_context


# ============= ENCODERS =============

class MLPEncoder(nn.Module):
    """Simple MLP encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2):
        super().__init__()
        mlp_layers = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(layers - 1):
            mlp_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        self.encoder = nn.Sequential(*mlp_layers)
    
    def forward(self, x):
        """(batch_size, input_dim) -> (batch_size, hidden_dim)"""
        return self.encoder(x)


class BERTEncoder(nn.Module):
    """BERT-like encoder with attention over input dimensions."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        layers: int = 2, 
        num_heads: int = 4, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(1, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.encoder_pooling = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """(batch_size, input_dim) -> (batch_size, hidden_dim)"""
        # Treat each dimension as a "token"
        tokens = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        projected_tokens = self.input_projection(tokens)  # (batch_size, input_dim, hidden_dim)
        
        # Apply transformer encoder
        encoded_tokens = self.transformer_encoder(projected_tokens)  # (batch_size, input_dim, hidden_dim)
        
        # Pool to single representation (mean pooling)
        pooled = encoded_tokens.mean(dim=1)  # (batch_size, hidden_dim)
        return self.encoder_pooling(pooled)  # (batch_size, hidden_dim)


# ============= DECODERS =============

class MLPDecoder(nn.Module):
    """Multi-layer MLP decoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2):
        super().__init__()
        mlp_layers = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(layers - 1):
            mlp_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        self.head = nn.Sequential(*mlp_layers)
    
    def forward(self, head_inputs):
        """(batch_size, output_dim, input_dim) -> (batch_size, output_dim, output_dim)"""
        # Vectorized MLP application
        batch_size, num_pos, input_dim = head_inputs.shape
        head_inputs_flat = head_inputs.reshape(-1, input_dim)
        outputs_flat = self.head(head_inputs_flat)
        return outputs_flat.reshape(batch_size, num_pos, -1)


class AttentionDecoder(nn.Module):
    """Multi-layer attention decoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head_projection = nn.Linear(input_dim, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.attention_head = nn.TransformerDecoder(decoder_layer, layers)
        self.head_output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, head_inputs):
        """(batch_size, output_dim, input_dim) -> (batch_size, output_dim, output_dim)"""
        # Project inputs and apply attention
        projected_inputs = self.head_projection(head_inputs)  # (batch_size, output_dim, hidden_dim)
        
        # Use projected inputs as both query and memory for self-attention
        attended = self.attention_head(projected_inputs, projected_inputs)
        return self.head_output(attended)


class MeanPooledDecoder(nn.Module):
    """Multi-layer mean-pooled decoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
    ):
        super().__init__()
        self.head_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack multiple mean-pooled layers for consistency with other decoders
        self.mean_pooled_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(layers):
            if i == layers - 1:  # Final layer outputs target dimension
                self.mean_pooled_layers.append(
                    MeanPooledMLP(current_dim, hidden_dim, output_dim, 1)
                )
            else:  # Intermediate layers maintain hidden dimension
                self.mean_pooled_layers.append(
                    MeanPooledMLP(current_dim, hidden_dim, hidden_dim, 1)
                )
                current_dim = hidden_dim
    
    def forward(self, head_inputs):
        """(batch_size, output_dim, input_dim) -> (batch_size, output_dim, output_dim)"""
        # Project inputs and apply multi-layer mean pooling
        x = self.head_projection(head_inputs)  # (batch_size, output_dim, hidden_dim)
        
        # Apply each mean-pooled layer sequentially
        for layer in self.mean_pooled_layers:
            x = layer(x)
        
        return x

class PositionalUNet(nn.Module):
    """
    Clean encoder-decoder architecture with separate, composable components.
    No if statements in forward pass!
    """
    
    def __init__(
        self,
        in_dims,
        hidden_dim,
        out_dim,
        encoder_type: Literal["mlp", "bert"] = "mlp",
        head_type: Literal["mlp", "attention", "mean_pooled"] = "mlp",
        encoder_layers: int = 2,
        head_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Convert Hydra configs to regular Python types
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        else:
            in_dims = list(in_dims)
            
        if hasattr(out_dim, '__iter__') and not isinstance(out_dim, int):
            out_dim = list(out_dim)
            self.out_shape = out_dim
            total_output_dim = 1
            for dim in out_dim:
                total_output_dim *= int(dim)
            self.output_dim = int(out_dim[0])  # First dimension determines positions
        else:
            self.out_shape = None
            total_output_dim = int(out_dim)
            self.output_dim = int(out_dim)
            
        self.in_dims = [int(dim) for dim in in_dims]
        self.hidden_dim = hidden_dim
        
        # Calculate dimensions from input specification
        # Assume inputs matching output_dim go to encoder, others are broadcast
        encoder_input_dim = sum(dim for dim in self.in_dims if dim == self.output_dim)
        broadcast_input_dim = sum(dim for dim in self.in_dims if dim != self.output_dim)
        
        # Position embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(self.output_dim, hidden_dim))
        
        # Create encoder during init
        if encoder_input_dim > 0:
            if encoder_type == "mlp":
                self.encoder = MLPEncoder(
                    input_dim=encoder_input_dim,
                    hidden_dim=hidden_dim,
                    layers=encoder_layers,
                )
            elif encoder_type == "bert":
                self.encoder = BERTEncoder(
                    input_dim=encoder_input_dim,
                    hidden_dim=hidden_dim,
                    layers=encoder_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
        else:
            self.encoder = None  # No encoder inputs
        
        # Create decoder during init  
        decoder_input_dim = hidden_dim + broadcast_input_dim + hidden_dim + encoder_input_dim // self.output_dim
        decoder_output_dim = total_output_dim // self.output_dim
        
        if head_type == "mlp":
            self.decoder = MLPDecoder(
                input_dim=decoder_input_dim,
                hidden_dim=hidden_dim,
                output_dim=decoder_output_dim,
                layers=head_layers,
            )
        elif head_type == "attention":
            self.decoder = AttentionDecoder(
                input_dim=decoder_input_dim,
                hidden_dim=hidden_dim,
                output_dim=decoder_output_dim,
                layers=head_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif head_type == "mean_pooled":
            self.decoder = MeanPooledDecoder(
                input_dim=decoder_input_dim,
                hidden_dim=hidden_dim,
                output_dim=decoder_output_dim,
                layers=head_layers,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        # Cached input mapping for efficiency (keys only, not dimensions)
        self._input_mapping_cached = False
        self._encoder_keys = []
        self._broadcast_keys = []
    
    def forward(self, **kwargs):
        """
        Clean forward pass: encode then decode. No if statements!
        """
        batch_size = next(iter(kwargs.values())).shape[0]
        
        # Cache input mapping on first forward pass for efficiency
        if not self._input_mapping_cached:
            for name, tensor in kwargs.items():
                if tensor.shape[-1] == self.output_dim:  # Data-dimension inputs (x_t, M_t, etc.)
                    self._encoder_keys.append(name)
                else:  # Other inputs (t, noise, etc.)
                    self._broadcast_keys.append(name)
            self._input_mapping_cached = True
        
        # Use cached mapping for efficient splitting
        encoder_tensors = [kwargs[key] for key in self._encoder_keys]
        broadcast_tensors = [kwargs[key] for key in self._broadcast_keys]
        
        # Concatenate encoder and broadcast inputs separately
        encoder_inputs = torch.cat(encoder_tensors, dim=-1) if encoder_tensors else torch.empty(batch_size, 0)
        broadcast_inputs = torch.cat(broadcast_tensors, dim=-1) if broadcast_tensors else torch.empty(batch_size, 0)
        
        # Verify dimensions match expectations (debug check)
        # This ensures our init-time calculations were correct
        
        # ENCODE: Create global context (polymorphic!)
        if encoder_inputs.shape[1] > 0 and self.encoder is not None:
            encoder_latent = self.encoder(encoder_inputs)  # (batch_size, hidden_dim)
        else:
            # No encoder inputs - create zero latent
            encoder_latent = torch.zeros(batch_size, self.hidden_dim, device=next(self.parameters()).device)
        
        # Prepare standardized decoder inputs
        encoder_latent_expanded = encoder_latent.unsqueeze(1).expand(-1, self.output_dim, -1)
        broadcast_expanded = broadcast_inputs.unsqueeze(1).expand(-1, self.output_dim, -1)
        pos_embeddings_expanded = self.pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Split encoder inputs per position - each position gets only its own values
        if encoder_inputs.shape[1] > 0:
            features_per_position = encoder_inputs.shape[1] // self.output_dim
            # Reshape: (batch_size, features_per_position, output_dim)
            pos_values_reshaped = encoder_inputs.view(batch_size, features_per_position, self.output_dim)
            # Transpose: (batch_size, output_dim, features_per_position)
            pos_values = pos_values_reshaped.transpose(1, 2)
        else:
            # No encoder inputs - create empty pos values
            pos_values = torch.empty(batch_size, self.output_dim, 0, device=next(self.parameters()).device)
        
        # Standardized head inputs for all decoder types
        head_inputs = torch.cat([
            encoder_latent_expanded,
            broadcast_expanded,
            pos_embeddings_expanded,
            pos_values
        ], dim=-1)
        
        # DECODE: Generate position-wise outputs (polymorphic!)
        output = self.decoder(head_inputs)  # (batch_size, output_dim, output_per_dim)
        
        # Reshape if needed
        if self.out_shape is not None:
            output = output.reshape(batch_size, *self.out_shape)
        else:
            output = output.reshape(batch_size, -1)
        
        return output 