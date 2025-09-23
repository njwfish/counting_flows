import torch
import torch.nn as nn
from enformer_pytorch import Enformer

from enformer_pytorch import from_pretrained


class SCFormer(nn.Module):
    def __init__(self, hidden_dim, noise_dim=100, layers=2, num_attn_layers=2, num_heads=4, class_dim=14, num_attn_proj_layers=2):
        super().__init__()
        enformer_dim = 3072
        seq_len = 896
        self.seq_len = seq_len
        self.enformer = from_pretrained('EleutherAI/enformer-official-rough')
        self.in_proj = nn.Sequential(
            nn.Linear(enformer_dim + noise_dim + 2 + class_dim, hidden_dim),
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

        self.in_attn_proj_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attn_proj_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
            for _ in range(num_attn_proj_layers)
        ])

        self.norm_proj_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_attn_proj_layers)
        ])

        self.out_proj_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, seq, noise, t, class_emb, target_sum=None):
        seq_emb = self.enformer(seq, return_only_embeddings=True).squeeze(0)
        seq_emb_rep = seq_emb.unsqueeze(0).expand(x_t.shape[0], -1, -1)
        t_rep = t.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        class_emb_rep = class_emb.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        noise_rep = noise.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        x_t_rep = x_t.unsqueeze(-1)
        x = torch.cat([seq_emb_rep, x_t_rep, t_rep, noise_rep, class_emb_rep], dim=-1)
        x = x_init = self.in_proj(x)
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            x, _ = attn(x, x, x)
            x = x + x_init
            x = norm(x)
        x_0_hat = torch.nn.functional.softplus(self.out_proj(x).squeeze(-1))
        if target_sum is not None:
            # because we have noise we will have target_sum for each noise sample which lets us check the number of noise samples
            m, _ = target_sum.shape
            b = x_t.shape[0] // m
            target_sum = target_sum.unsqueeze(0).expand(b, -1, -1).reshape(b * m, -1)
            y = torch.cat([x_0_hat.unsqueeze(-1), x_t.unsqueeze(-1), target_sum.unsqueeze(-1), x], dim=-1).reshape(b, m, self.seq_len, -1)
            y = y.reshape(b, m * self.seq_len, -1)
            y = y_init = self.in_attn_proj_proj(y)
            for attn_proj, norm in zip(self.attn_proj_layers, self.norm_proj_layers):
                y, _ = attn_proj(y, y, y)
                y = y + y_init
                y = norm(y)
            y = y.reshape(b * m, self.seq_len, -1)
            y = self.out_proj_proj(y).squeeze(-1) + x_0_hat
            x_0_hat = torch.nn.functional.softplus(y)
        return x_0_hat


class SCFormerMSE(nn.Module):
    def __init__(self, hidden_dim, layers=2, num_attn_layers=2, num_heads=4, class_dim=14):
        super().__init__()
        enformer_dim = 3072
        seq_len = 896
        self.seq_len = seq_len
        self.enformer = from_pretrained('EleutherAI/enformer-official-rough')
        self.in_proj = nn.Sequential(
            nn.Linear(enformer_dim + class_dim, hidden_dim),
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

    def forward(self, x_t, seq, t, class_emb, target_sum=None):
        seq_emb = self.enformer(seq, return_only_embeddings=True).squeeze(0)
        seq_emb_rep = seq_emb.unsqueeze(0).expand(class_emb.shape[0], -1, -1)
        class_emb_rep = class_emb.unsqueeze(1).expand(-1, seq_emb_rep.shape[1], -1)
        # print(seq_emb_rep.shape, class_emb_rep.shape)
        x = torch.cat([seq_emb_rep, class_emb_rep], dim=-1)
        x = x_init = self.in_proj(x)
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            x, _ = attn(x, x, x)
            x = x + x_init
            x = norm(x)
        x_0_hat = torch.nn.functional.softplus(self.out_proj(x).squeeze(-1))
        return x_0_hat