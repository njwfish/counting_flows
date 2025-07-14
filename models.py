import torch
import torch.nn as nn

class EnergyScorePosterior(nn.Module):
    """
    Distributional-diffusion energy score (eq.14) with m-sample approximation.
    f_θ(x_t, t, z, ε) → x₀̂ = x_t + Δx
    """
    def __init__(
        self,
        x_dim: int,
        context_dim: int = 0,
        hidden: int = 128,
        noise_dim: int = None,
        sigma: float = 1.0,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
    ):
        super().__init__()
        # dimension of your Gaussian noise
        self.noise_dim      = noise_dim if noise_dim is not None else hidden
        # # samples per data point
        self.m              = m_samples
        # λ weight for the interaction term
        self.lambda_energy  = lambda_energy

        # total #features going into the MLP:
        #  - x_t: x_dim
        #  - t:     1
        #  - z: context_dim
        #  - ε: noise_dim
        self._in_dim = x_dim * 2 + 1 + context_dim + self.noise_dim

        # build your network
        self.net = nn.Sequential(
            nn.Linear(self._in_dim, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, x_dim),
        )


    def forward(self, x_t, M_t, t, z=None, noise=None):
        """
        x_t: [B, x_dim]
        z:   [B, context_dim]
        t:   [B] or [B,1]
        noise: [B, noise_dim] or None (will be sampled)
        → returns [B, x_dim]
        """
        B = x_t.shape[0]
        # sample noise if needed
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=x_t.device)

        # unify t to shape [B,1]
        if t.dim() == 1:
            t_col = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[1] == 1:
            t_col = t
        else:
            raise ValueError(f"t must be [B] or [B,1], got {tuple(t.shape)}")

        # concat everything
        inp = torch.cat([x_t.float(), t_col.float(), z.float() if z is not None else torch.zeros(B, 0, device=x_t.device), M_t.float(), noise], dim=-1)

        # sanity check
        assert inp.shape[1] == self._in_dim, (
            f"got inp dim={inp.shape[1]}, expected {self._in_dim}"
        )

        # predict Δx, add to x_t
        return self.net(inp) + x_t

    def sample(self, x_t, M_t, t, z=None):
        x0_hat = self.forward(x_t, M_t, t, z)
        return x0_hat.round().long()

    def loss(self, x0_true, x_t, M_t, t, z=None):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||x0_trueᵢ - x̂ᵢⱼ||
            - (λ/(2(m-1))) ∑_{j≠j'} ||x̂ᵢⱼ - x̂ᵢⱼ'||
          ]
        """
        n, m, λ = x0_true.size(0), self.m, self.lambda_energy

        # replicate each input m times 
        x_t_rep = x_t.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        z_rep   =   z.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1) if z is not None else None
        M_t_rep = M_t.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        # flatten t to 1D so forward() will turn it into [n*m,1] (for t)
        if t.dim() == 2 and t.shape[1] == 1:
            t_rep = t.expand(-1, m).reshape(n * m)
        elif t.dim() == 1:
            t_rep = t.unsqueeze(1).expand(-1, m).reshape(n * m)
        else:
            raise ValueError(f"t must be [n] or [n,1], got {tuple(t.shape)}")

        # sample m·n noises (for noise)
        noise = torch.randn(n * m, self.noise_dim, device=x_t.device)

        # get all x̂ preds: [n*m, x_dim] → view [n, m, x_dim] (for x0_preds)
        x0_preds = self.forward(x_t_rep, M_t_rep, t_rep, z_rep, noise)
        x0_preds = x0_preds.view(n, m, -1)  # [n, m, d]

        # confinement
        x0_true_rep = x0_true.unsqueeze(1).expand(-1, m, -1)
        term_conf   = (x0_preds - x0_true_rep).norm(dim=2).mean(dim=1)  # [n]

        # interaction 
        pdists    = torch.stack([torch.pdist(x0_preds[i], p=2) for i in range(n)], dim=0)
        mean_pd   = pdists.mean(dim=1)                                # [n]
        term_int  = (λ / 2.0) * mean_pd                               # [n]

        return (term_conf - term_int).mean(), x0_preds.mean(dim=1)
