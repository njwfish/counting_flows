import torch
import torch.nn as nn
from enformer_pytorch import Enformer

from enformer_pytorch import from_pretrained


class SCFormer(nn.Module):
    def __init__(self, hidden_dim, noise_dim=100, layers=2, num_attn_layers=2, num_heads=4):
        super().__init__()
        enformer_dim = 3072
        self.enformer = from_pretrained('EleutherAI/enformer-official-rough')
        self.in_proj = nn.Sequential(
            nn.Linear(enformer_dim + noise_dim + 2, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_attn_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_attn_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, z, noise, t):
        z = self.enformer(z, return_only_embeddings=True)
        z = z.squeeze(0).unsqueeze(0).expand(x_t.shape[0], -1, -1)
        t = t.unsqueeze(-1).expand(-1, x_t.shape[1], -1)
        x_t = x_t.unsqueeze(-1)
        noise = noise.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        x = torch.cat([z, x_t, t, noise], dim=-1)
        x = self.in_proj(x)
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            x, _ = attn(x, x, x)
            x = norm(x)
        x = self.out_proj(x).squeeze(-1)
        return x
        
    