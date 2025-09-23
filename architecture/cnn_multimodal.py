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
    Standard ResNet-style residual block - pure image feature extraction.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(min(8, out_channels), out_channels)  # Fewer groups
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.selu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.selu(out)
        
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
        
        # Time embedding - more efficient
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        
        # Noise embedding - more efficient
        self.noise_embed = nn.Sequential(
            nn.Linear(noise_dim, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim),
        )
        
        # Removed count_global_embed - not needed anymore
        
        # Initial conv - smaller kernel to reduce parameters
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True)
        self.init_gn = nn.GroupNorm(min(8, base_channels), base_channels)  # Fewer groups
        
        # Encoder blocks (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i, multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * multiplier
            
            # Downsampling if not first block
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 1, stride=2, bias=True),
                    nn.GroupNorm(min(8, out_channels), out_channels),
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
                            stride=stride, downsample=downsample
                        )
                    )
                else:
                    # Subsequent blocks maintain channels
                    level_blocks.append(
                        ResidualBlock(out_channels, out_channels)
                    )
                stride = 1  # Only first block can downsample
                downsample = None
                current_channels = out_channels
            
            self.encoder_blocks.append(level_blocks)
            
        # Middle blocks (bottleneck)
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(current_channels, current_channels)
            for _ in range(num_res_blocks)
        ])
        
        # Time/noise/class conditioning injection points
        self.time_projections = nn.ModuleList([
            nn.Linear(time_embed_dim, base_channels * mult) 
            for mult in channel_multipliers
        ])
        
        # Image feature aggregation at multiple scales - more efficient
        self.feature_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_channels * mult, time_embed_dim // 8),  # Even smaller
                nn.SiLU()
            ) for mult in channel_multipliers
        ])
        
        # Simple fusion - more parameter efficient
        total_img_features = len(channel_multipliers) * (time_embed_dim // 8)
        self.image_feature_fusion = nn.Sequential(
            nn.Linear(total_img_features, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim)
        )
        
        # Count embedding network - more efficient for high-dim counts
        self.count_embed = nn.Sequential(
            nn.Linear(count_dim, time_embed_dim // 2),  # Bottleneck first
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim),
        )
        
        # Final count prediction network - streamlined but expressive
        self.count_predictor = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),  # *2 for concat
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
        
        # === COMBINE CONDITIONING SIGNALS FIRST ===
        # Time embedding
        if t.dim() == 2:
            t = t.squeeze(-1)
        time_emb = self.time_embed(t)  # [B, time_embed_dim]
        
        # === ENCODE IMAGES FOR CONDITIONING ===
        # Initial convolution
        x = F.selu(self.init_gn(self.init_conv(img)))
        
        # Extract multi-scale image features efficiently
        image_features = []
        
        # Encoder path - extract features at each resolution with time conditioning
        for level_idx, level_blocks in enumerate(self.encoder_blocks):
            # Process through residual blocks
            for block in level_blocks:
                x = block(x)
                
            # Add time conditioning at this scale (finally use time_projections!)
            time_proj = self.time_projections[level_idx](time_emb)  # [B, channels]
            time_proj = time_proj.unsqueeze(-1).unsqueeze(-1)  # [B, channels, 1, 1]
            x = x + time_proj
                
            # Aggregate features at this scale
            level_features = self.feature_aggregators[level_idx](x)  # [B, time_embed_dim//4]
            image_features.append(level_features)
        
        # Middle blocks for final feature refinement
        for block in self.middle_blocks:
            x = block(x)
        
        # Noise embedding (always provided)
        noise_emb = self.noise_embed(noise)  # [B, time_embed_dim]
        
        # Count embedding
        count_emb = self.count_embed(x_t)  # [B, time_embed_dim]
        
        # === EFFICIENT FEATURE FUSION ===
        # Concatenate all image features and fuse
        concatenated_img_features = torch.cat(image_features, dim=1)  # [B, total_img_features]
        fused_img_features = self.image_feature_fusion(concatenated_img_features)  # [B, time_embed_dim]
        
        # Combine all conditioning: count + time + noise + image
        global_conditioning = time_emb + noise_emb  # [B, time_embed_dim]
        conditioned_count_emb = count_emb + global_conditioning  # [B, time_embed_dim]
        
        # === PREDICT DENOISED COUNTS ===
        # Simple concatenation fusion - much faster than cross-attention
        fused_features = torch.cat([conditioned_count_emb, fused_img_features], dim=1)  # [B, time_embed_dim * 2]
        
        # Predict what the denoised counts should be
        predicted_counts = self.count_predictor(fused_features)  # [B, count_dim]
        predicted_counts = torch.nn.functional.softplus(predicted_counts + x_t)
        
        return predicted_counts


# Factory functions for different scales - more parameter efficient
def create_cnn_count_diffusion_256(count_dim: int = 650, **kwargs):
    """Create efficient CNN count diffusion model for 256x256 images."""
    return CNNCountDiffusion(
        img_size=256,
        base_channels=48,  # Reduced from 64
        channel_multipliers=(1, 2, 4, 8),  # Removed 16x multiplier
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=512,  # Reduced from 1024
        **kwargs
    )

def create_cnn_count_diffusion_128(count_dim: int = 650, **kwargs):
    """Create efficient CNN count diffusion model for 128x128 images.""" 
    return CNNCountDiffusion(
        img_size=128,
        base_channels=40,  # Reduced from 64
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=384,  # Reduced from 768
        **kwargs
    )

def create_cnn_count_diffusion_64(count_dim: int = 650, **kwargs):
    """Create efficient CNN count diffusion model for 64x64 images."""
    return CNNCountDiffusion(
        img_size=64,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 6),  # Slightly reduced
        num_res_blocks=2,
        count_dim=count_dim,
        time_embed_dim=256,  # Reduced from 512
        **kwargs
    )
