import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class CountConditioningModule(nn.Module):
    """
    Module to condition CNN features on count vectors using FiLM (Feature-wise Linear Modulation).
    
    This takes count vectors and generates scale and shift parameters to modulate
    the intermediate CNN feature maps.
    """
    def __init__(self, count_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.count_dim = count_dim
        self.feature_dim = feature_dim
        
        # Process count vector to conditioning parameters
        self.count_processor = nn.Sequential(
            nn.Linear(count_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2)  # *2 for scale and shift
        )
        
    def forward(self, feature_maps: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Apply count conditioning to feature maps.
        
        Args:
            feature_maps: [B, C, H, W] CNN feature maps
            counts: [B, count_dim] count vectors
            
        Returns:
            conditioned_features: [B, C, H, W] modulated feature maps
        """
        # Generate scale and shift parameters
        conditioning = self.count_processor(counts)  # [B, feature_dim * 2]
        scale, shift = conditioning.chunk(2, dim=1)  # Each [B, feature_dim]
        
        # Reshape for broadcasting
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1]
        
        # Apply FiLM conditioning
        return feature_maps * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """
    Standard ResNet-style residual block with optional count conditioning.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None,
        count_dim: Optional[int] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # Optional count conditioning
        self.count_conditioning = None
        if count_dim is not None:
            self.count_conditioning = CountConditioningModule(count_dim, out_channels)
    
    def forward(self, x: torch.Tensor, counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # Apply count conditioning if available
        if self.count_conditioning is not None and counts is not None:
            out = self.count_conditioning(out, counts)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out, inplace=True)
        
        return out


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding similar to the one used in diffusion models.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] or [B, 1] time steps
        Returns:
            embeddings: [B, embed_dim] time embeddings
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(1)
            
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
            
        return emb


class CNNCountDiffusion(nn.Module):
    """
    CNN-based count diffusion model that uses images as rich conditioning.
    
    This is a COUNT DIFFUSION MODEL where:
    - Images are processed through a CNN to extract rich hierarchical features
    - These image features condition the count prediction at multiple scales
    - The model predicts denoised counts (x_0) given noisy counts (x_t) and images
    
    Architecture:
    1. CNN image encoder extracts multi-scale features
    2. Image features are integrated with count embeddings via cross-attention/fusion
    3. Time and noise embeddings provide additional conditioning
    4. Final MLP predicts the denoised count vector
    """
    
    def __init__(
        self,
        # Image parameters
        img_size: int = 256,
        in_channels: int = 1,
        # Count parameters
        count_dim: int = 10,
        # Noise parameters  
        noise_dim: int = 16,
        # Architecture parameters
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_embed_dim: int = 512,
        # Optional parameters
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.count_dim = count_dim
        self.noise_dim = noise_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Noise embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(noise_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Count preprocessing - project to time embed dimension for global conditioning
        self.count_global_embed = nn.Sequential(
            nn.Linear(count_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(base_channels)
        
        # Encoder blocks (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i, multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * multiplier
            
            # Downsampling if not first block
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 1, stride=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                stride = 2
            else:
                downsample = None
                stride = 1
            
            # Add residual blocks for this level
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                if j == 0:
                    # First block handles potential downsampling and channel change
                    level_blocks.append(
                        ResidualBlock(
                            current_channels, out_channels, 
                            stride=stride, downsample=downsample,
                            count_dim=count_dim
                        )
                    )
                else:
                    # Subsequent blocks maintain channels
                    level_blocks.append(
                        ResidualBlock(out_channels, out_channels, count_dim=count_dim)
                    )
                stride = 1  # Only first block can downsample
                downsample = None
                current_channels = out_channels
            
            self.encoder_blocks.append(level_blocks)
            
        # Middle blocks (bottleneck)
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, count_dim=count_dim)
            for _ in range(num_res_blocks)
        ])
        
        # Time/noise/class conditioning injection points
        self.time_projections = nn.ModuleList([
            nn.Linear(time_embed_dim, base_channels * mult) 
            for mult in channel_multipliers
        ])
        
        # Image feature aggregation at multiple scales for count conditioning
        self.feature_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_channels * mult, time_embed_dim),
                nn.SiLU()
            ) for mult in channel_multipliers
        ])
        
        # Cross-attention module for integrating image features with count embeddings
        self.count_cross_attention = nn.MultiheadAttention(
            embed_dim=time_embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Count embedding network
        self.count_embed = nn.Sequential(
            nn.Linear(count_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Final count prediction network - optimized for high-dimensional counts (~650)
        self.count_predictor = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),  # *2 for concat of count + image features
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(time_embed_dim, count_dim),
            nn.Softplus()  # Ensure positive counts output
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor,
        img: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for count diffusion model.
        
        Args:
            x_t: [B, count_dim] noisy count vectors (the main input we're denoising)
            t: [B] or [B, 1] timesteps
            noise: [B, noise_dim] noise vectors for conditioning (required)
            img: [B, C, H, W] conditioning images (required)
            
        Returns:
            [B, count_dim] predicted denoised counts (positive values)
        """
        # Both img and noise are now required parameters
        
        B = x_t.shape[0]
        
        # === ENCODE IMAGES FOR CONDITIONING ===
        # Initial convolution
        x = F.relu(self.init_bn(self.init_conv(img)), inplace=True)
        
        # Extract multi-scale image features
        image_features = []
        
        # Encoder path - extract features at each resolution
        for level_idx, level_blocks in enumerate(self.encoder_blocks):
            # Process through residual blocks  
            for block in level_blocks:
                x = block(x)  # No count conditioning in encoder - pure image features
                
            # Aggregate features at this scale
            level_features = self.feature_aggregators[level_idx](x)  # [B, time_embed_dim]
            image_features.append(level_features)
        
        # Middle blocks for final feature refinement
        for block in self.middle_blocks:
            x = block(x)
        
        # Final feature aggregation
        final_features = self.feature_aggregators[-1](x)  # [B, time_embed_dim] 
        image_features.append(final_features)
        
        # === COMBINE CONDITIONING SIGNALS ===
        # Time embedding
        if t.dim() == 2:
            t = t.squeeze(-1)
        time_emb = self.time_embed(t)  # [B, time_embed_dim]
        
        # Noise embedding (always provided)
        noise_emb = self.noise_embed(noise)  # [B, time_embed_dim]
        
        # No class embedding needed
        
        # Count embedding
        count_emb = self.count_embed(x_t)  # [B, time_embed_dim]
        
        # Combine all conditioning into global context
        global_conditioning = time_emb + noise_emb  # [B, time_embed_dim]
        
        # === FUSE IMAGE FEATURES WITH COUNT INFORMATION ===
        # Stack image features from different scales
        stacked_img_features = torch.stack(image_features, dim=1)  # [B, num_scales, time_embed_dim]
        
        # Add global conditioning to count embedding
        conditioned_count_emb = count_emb + global_conditioning  # [B, time_embed_dim]
        conditioned_count_emb = conditioned_count_emb.unsqueeze(1)  # [B, 1, time_embed_dim]
        
        # Cross-attention: let count embedding attend to image features
        attended_features, _ = self.count_cross_attention(
            query=conditioned_count_emb,  # [B, 1, time_embed_dim]
            key=stacked_img_features,     # [B, num_scales, time_embed_dim] 
            value=stacked_img_features    # [B, num_scales, time_embed_dim]
        )
        attended_features = attended_features.squeeze(1)  # [B, time_embed_dim]
        
        # === PREDICT DENOISED COUNTS ===
        # Combine attended image features with original count embedding
        fused_features = torch.cat([conditioned_count_emb.squeeze(1), attended_features], dim=1)  # [B, time_embed_dim * 2]
        
        # Predict what the denoised counts should be
        predicted_counts = self.count_predictor(fused_features)  # [B, count_dim]
        
        return predicted_counts


# Factory functions for different scales
def create_cnn_count_diffusion_256(count_dim: int = 650, **kwargs):
    """Create CNN count diffusion model optimized for 256x256 images and high-dimensional counts."""
    return CNNCountDiffusion(
        img_size=256,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8, 16),
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=1024,  # Larger embedding for high count dim
        **kwargs
    )

def create_cnn_count_diffusion_128(count_dim: int = 650, **kwargs):
    """Create CNN count diffusion model for 128x128 images and high-dimensional counts.""" 
    return CNNCountDiffusion(
        img_size=128,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=768,  # Larger embedding for high count dim
        **kwargs
    )

def create_cnn_count_diffusion_64(count_dim: int = 650, **kwargs):
    """Create CNN count diffusion model for 64x64 images and high-dimensional counts."""
    return CNNCountDiffusion(
        img_size=64,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=512,  # Smaller embedding for lighter model
        **kwargs
    )
