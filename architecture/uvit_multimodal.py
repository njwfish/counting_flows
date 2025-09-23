import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
from .uvit import timestep_embedding, patchify, unpatchify, Attention, Block, PatchEmbed
import einops


class CountPatchEmbedding(nn.Module):
    """
    Convert count vectors into patch-like representations for attention.
    
    Takes a count vector [B, count_dim] and converts it to [B, num_count_patches, embed_dim]
    to be compatible with image patches in the attention mechanism.
    """
    def __init__(self, count_dim, embed_dim, num_count_patches=4):
        super().__init__()
        self.count_dim = count_dim
        self.num_count_patches = num_count_patches
        self.embed_dim = embed_dim
        
        # Project count vector to higher dimension
        self.count_projection = nn.Sequential(
            nn.Linear(count_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim * num_count_patches),
        )
        
        # Learnable position embeddings for count patches
        self.count_pos_embed = nn.Parameter(torch.zeros(1, num_count_patches, embed_dim))
        trunc_normal_(self.count_pos_embed, std=.02)
        
    def forward(self, counts):
        """
        Args:
            counts: [B, count_dim] count vectors
        Returns:
            count_patches: [B, num_count_patches, embed_dim] patch representations
        """
        B = counts.shape[0]
        
        # Project to patch representations
        count_features = self.count_projection(counts)  # [B, embed_dim * num_count_patches]
        count_patches = count_features.view(B, self.num_count_patches, self.embed_dim)  # [B, num_count_patches, embed_dim]
        
        # Add positional embeddings
        count_patches = count_patches + self.count_pos_embed
        
        return count_patches


class CountDecoder(nn.Module):
    """
    Decode count patches back to count vectors.
    """
    def __init__(self, embed_dim, count_dim, num_count_patches=4):
        super().__init__()
        self.num_count_patches = num_count_patches
        
        # Project from patches back to count vector
        self.count_decoder = nn.Sequential(
            nn.Linear(embed_dim * num_count_patches, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, count_dim),
            nn.Softplus()
        )
        
    def forward(self, count_patches):
        """
        Args:
            count_patches: [B, num_count_patches, embed_dim] 
        Returns:
            counts: [B, count_dim] reconstructed count vectors
        """
        B = count_patches.shape[0]
        
        # Flatten patches and decode
        count_features = count_patches.view(B, -1)  # [B, embed_dim * num_count_patches]
        counts = self.count_decoder(count_features)  # [B, count_dim]
        
        return counts


class MultimodalUViT(nn.Module):
    """
    Multimodal U-ViT that processes both images and count vectors.
    
    Architecture:
    1. Images -> PatchEmbed -> image patches [B, num_img_patches, embed_dim]
    2. Counts -> CountPatchEmbedding -> count patches [B, num_count_patches, embed_dim]  
    3. Concatenate all patches along sequence dimension
    4. Add time and optional label embeddings
    5. Process through U-Net style transformer blocks
    6. Split patches back to image and count components
    7. Decode to image and count outputs
    """
    
    def __init__(
        self, 
        # Image parameters
        img_size=28, patch_size=4, in_chans=1, 
        # Count parameters  
        count_dim=10, num_count_patches=4,
        # Noise parameters
        noise_dim=16,
        # Architecture parameters
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
        qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
        mlp_time_embed=False, num_classes=-1,
        use_checkpoint=False, conv=True, skip=True, time_dim=2
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.count_dim = count_dim
        self.num_count_patches = num_count_patches
        self.noise_dim = noise_dim

        # Image processing
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_img_patches = (img_size // patch_size) ** 2
        
        # Count processing
        self.count_embed = CountPatchEmbedding(count_dim, embed_dim, num_count_patches)
        
        # Total number of patches
        total_patches = num_img_patches + num_count_patches

        # Time embedding
        if mlp_time_embed:
            self.time_embed = nn.Sequential(
                nn.Linear(time_dim * embed_dim, 4 * embed_dim),
                nn.SiLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
        elif time_dim > 1:
            self.time_embed = nn.Linear(time_dim * embed_dim, embed_dim)
        else:
            self.time_embed = nn.Identity()
        
        # Noise embedding (for energy score / distributional diffusion)
        self.noise_proj = nn.Linear(noise_dim, embed_dim)
        self.noise_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Optional class embedding
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 3  # time + noise + label
        else:
            self.extras = 2  # time + noise

        # Positional embeddings for all patches + extra tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + total_patches, embed_dim))

        # Transformer blocks (U-Net style)
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        
        # Separate decoders for each modality
        # Image decoder
        self.patch_dim = patch_size ** 2 * in_chans
        self.img_decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.img_final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()
        
        # Count decoder
        self.count_decoder = CountDecoder(embed_dim, count_dim, num_count_patches)
        
        # Store patch counts for splitting
        self.num_img_patches = num_img_patches
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'count_embed.count_pos_embed'}

    def forward(self, x_t, t, y=None, noise=None, img=None):
        """
        Forward pass for multimodal inputs.
        
        Args:
            x_t: Dict with keys 'img' and 'counts'
                - x_t['img']: [B, C, H, W] noisy images
                - x_t['counts']: [B, count_dim] noisy count vectors
            t: [B, 1] time steps
            y: [B] optional class labels
            noise: [B, noise_dim] optional noise vector (for energy score)
            
        Returns:
            Dict with keys 'img' and 'counts' containing predictions
        """ 
        # Extract modalities
        if img is not None:
            img_t = img # [B, C, H, W]
            counts_t = x_t # [B, count_dim]
        else:
            img_t = x_t['img']
            counts_t = x_t['counts']
        
        B = img_t.shape[0]
        
        # Process images to patches
        img_patches = self.patch_embed(img_t)  # [B, num_img_patches, embed_dim]
        
        # Process counts to patches
        count_patches = self.count_embed(counts_t)  # [B, num_count_patches, embed_dim]
        
        # Concatenate all patches
        x = torch.cat([img_patches, count_patches], dim=1)  # [B, total_patches, embed_dim]
        
        # Add time embedding
        if isinstance(t, dict):
            t = torch.hstack([t[k] for k in t])

        t_flat = t.flatten()
        time_token = self.time_embed(timestep_embedding(t_flat, self.embed_dim).reshape(B, -1)).unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)  # [B, 1 + total_patches, embed_dim]
        
        # Project noise to embed_dim and then embed
        noise_proj = self.noise_proj(noise)  # [B, embed_dim]
        noise_token = self.noise_embed(noise_proj)
        noise_token = noise_token.unsqueeze(dim=1)  # [B, 1, embed_dim]
        x = torch.cat((noise_token, x), dim=1)  # [B, 2 + total_patches, embed_dim]
        
        # Add optional class embedding
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)  # [B, 1, embed_dim]
            x = torch.cat((label_emb, x), dim=1)  # [B, extras + total_patches, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Process through transformer blocks (U-Net style)
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        
        # Remove extra tokens (time, optional label)
        x = x[:, self.extras:, :]  # [B, total_patches, embed_dim]
        
        # Split back into modalities
        img_patches_out = x[:, :self.num_img_patches, :]  # [B, num_img_patches, embed_dim]
        count_patches_out = x[:, self.num_img_patches:, :]  # [B, num_count_patches, embed_dim]
        
        # Decode images
        img_patches_decoded = self.img_decoder_pred(img_patches_out)  # [B, num_img_patches, patch_dim]
        img_out = unpatchify(img_patches_decoded, self.in_chans)  # [B, C, H, W]
        img_out = self.img_final_layer(img_out)
        
        # Decode counts
        counts_out = self.count_decoder(count_patches_out)  # [B, count_dim]
        
        # this returns counts only if img is None
        if img is not None:
            return counts_out

        return {'img': img_out, 'counts': counts_out}

