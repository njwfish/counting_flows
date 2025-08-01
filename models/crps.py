import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, device, eps=1e-10):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, tau, straight_through=False):
    """
    logits: [B, D, V]
    Returns relaxed one-hot [B, D, V]; if straight_through=True, returns a
    tensor that is hard in forward but uses soft for backward.
    """
    gumbel = sample_gumbel(logits.shape, device=logits.device)
    y = (logits + gumbel) / tau
    soft = F.softmax(y, dim=-1)  # [B, D, V]

    if straight_through:
        index = soft.argmax(dim=-1, keepdim=True)  # [B, D, 1]
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, index, 1.0)
        # forward uses hard, backward uses soft
        return hard + (soft - soft.detach())
    else:
        return soft


class RelaxedEnergyScoreLoss(nn.Module):
    """
    Gumbel-Softmax relaxed multivariate energy score (differentiable surrogate).
    Mirrors the interface of EnergyScoreLoss (eq.14 style) but works on relaxed
    joint vectors constructed via Gumbel-Softmax.
    """
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,  # kept for interface compatibility but not used in strict ES
        tau: float = 1.0,
        tau_min: float = 0.1,
        anneal_steps: int = 0,
        norm: str = "l2",
        straight_through: bool = True,
        min_value: int = 0,
        value_range: int = 10,
        lambda_int: float = 1.0,
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy  # unused in canonical relaxed ES, kept for API
        self.tau_init = tau
        self.tau_min = tau_min
        self.anneal_steps = anneal_steps
        self.norm = norm
        self.straight_through = straight_through
        self.min_value = min_value
        self.value_range = value_range
        self.V = value_range

        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        # value vector for expected count per dimension
        self.register_buffer("v_tensor", torch.arange(self.V, dtype=torch.float32))  # (V,)

    def _current_tau(self):
        step = self.step.item()
        if step >= self.anneal_steps:
            return self.tau_min
        frac = step / max(1, self.anneal_steps)
        return self.tau_init + (self.tau_min - self.tau_init) * frac

    def forward(self, inputs):
        """
        Single noisy forward pass, returns logits [B, D, V].
        """
        inputs_with_noise = inputs.copy()
        base_input = list(inputs.values())[0]
        batch_size = base_input.shape[0]
        inputs_with_noise["noise"] = torch.randn(batch_size, self.noise_dim, device=base_input.device)
        return self.architecture(**inputs_with_noise)

    def sample(self, **kwargs):
        """
        Produce a discrete joint sample using Gumbel-Softmax straight-through
        from one noise draw.
        """
        base_input = list(kwargs.values())[0]
        B = base_input.shape[0]
        device = base_input.device
        tau = self._current_tau()

        z = torch.randn(B, self.noise_dim, device=device)
        logits = self.architecture(**{**kwargs, "noise": z})  # [B, D, V]
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)  # [B,1,V]
        # straight-through to get discrete-style
        soft = gumbel_softmax_sample(logits, tau, straight_through=True)  # [B,D,V]
        sample = soft.argmax(dim=-1)  # [B,D]
        return sample.clamp(min=self.min_value, max=self.min_value + self.value_range)

    def loss(self, target, inputs):
        """
        Relaxed energy score on Gumbel-Softmax proxies:
          L = mean_i [ (1/m)∑_j ||X̃_{ij}-y_i||  -  (1 / (2 m (m-1))) ∑_{j≠j'} ||X̃_{ij}-X̃_{ij'}|| ]
        """
        B = target.shape[0]
        device = target.device

        # advance annealing step
        self.step += 1
        tau = self._current_tau()

        # prepare target float vector
        y = (target - self.min_value).clamp(min=0, max=self.value_range).float()  # [B, D]

        # collect m relaxed joint vectors: each is [B, D]
        X_relaxed = []
        for _ in range(self.m):
            z = torch.randn(B, self.noise_dim, device=device)
            logits = self.architecture(**{**inputs, "noise": z})  # [B, D, V] expected
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)  # [B,1,V]
            soft = gumbel_softmax_sample(logits, tau, straight_through=self.straight_through)  # [B,D,V]
            exp_vec = (soft * self.v_tensor.view(1, 1, -1)).sum(-1)  # [B, D]
            X_relaxed.append(exp_vec)
        X = torch.stack(X_relaxed, dim=0)  # [m, B, D]

        # choose distance
        if self.norm == "l2":
            def dist(a, b): return (a - b).norm(dim=-1)  # [*,B]
        elif self.norm == "l1":
            def dist(a, b): return (a - b).abs().sum(dim=-1)
        else:
            raise ValueError("Unsupported norm")

        # confinement: (1/m) Σ_j ||X_j - y||
        term_conf = dist(X, y.unsqueeze(0)).mean(0)  # [B]

        # interaction: (1 / (2 m (m-1))) Σ_{j≠j'} ||X_j - X_j'||
        diff = dist(X.unsqueeze(0), X.unsqueeze(1))  # [m, m, B]
        m = self.m
        # upper triangle without diagonal then symmetrize
        mask = torch.triu(torch.ones(m, m, device=device), 1)
        pairwise = (diff * (mask + mask.T).unsqueeze(-1)).sum((0, 1))  # [B]
        term_int = pairwise / (m * (m - 1))  # [B]

        es_relaxed = term_conf - 0.5 * term_int  # [B]
        return es_relaxed.mean()
