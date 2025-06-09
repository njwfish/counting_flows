"""
Neural Network Models for Count-based Flow Matching

Provides different neural architectures for predicting count distributions:
- NBPosterior: Negative Binomial parameters (r, p)
- BetaBinomialPosterior: Beta-Binomial parameters (n, alpha, beta)  
- MLERegressor: Direct count prediction via log(1 + x₀)
- ZeroInflatedPoissonPosterior: Zero-Inflated Poisson parameters (λ, π)
- IQNPosterior: Implicit Quantile Networks for count prediction
- MMDPosterior: Maximum Mean Discrepancy with L2 kernel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Beta
from abc import ABC, abstractmethod
import math

class BetaBinomial:
    def __init__(self, total_count, concentration1, concentration0):
        self.total_count = total_count
        self.concentration1 = concentration1  # alpha
        self.concentration0 = concentration0  # beta
        
    def sample(self):
        # Sample p from Beta(alpha, beta), then sample from Binomial(n, p)
        beta_dist = Beta(self.concentration1, self.concentration0)
        p = beta_dist.sample()
        binomial_dist = torch.distributions.Binomial(total_count=self.total_count, probs=p)
        return binomial_dist.sample()
    
    def log_prob(self, value):
        # Beta-binomial log-likelihood
        n = torch.tensor(self.total_count)
        alpha = self.concentration1  
        beta = self.concentration0
        k = value
        
        # Log-likelihood: log(C(n,k)) + log(B(k+alpha, n-k+beta)) - log(B(alpha, beta))
        log_comb = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        log_beta_num = torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) - torch.lgamma(n + alpha + beta)
        log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        
        return log_comb + log_beta_num - log_beta_den
    
    @property
    def mean(self):
        # Mean of beta-binomial: n * alpha / (alpha + beta)
        return self.total_count * self.concentration1 / (self.concentration1 + self.concentration0)


class BaseCountModel(nn.Module, ABC):
    """Base class for count prediction models"""
    
    def __init__(self, x_dim, context_dim, hidden=128):
        super().__init__()
        self.x_dim = x_dim
        self.context_dim = context_dim
        
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1 + context_dim, hidden),
            nn.SELU(), 
            nn.Linear(hidden, hidden), 
            nn.SELU(),
            nn.Linear(hidden, hidden), 
            nn.SELU(),
            nn.Linear(hidden, self.output_dim)
        )
    
    @property
    @abstractmethod
    def output_dim(self):
        """Number of output dimensions"""
        pass
    
    def forward(self, x_t, z, t):
        """Common forward pass"""
        t = t.expand_as(x_t[:, :1])
        inp = torch.cat([x_t.float(), t.float(), z.float()], dim=-1)
        h = self.net(inp)
        return self.process_output(h)
    
    @abstractmethod
    def process_output(self, h):
        """Process raw network output into model-specific parameters"""
        pass
    
    @abstractmethod
    def sample(self, x_t, z, t, use_mean=False):
        """Sample x0 predictions - unified interface"""
        pass
    
    @abstractmethod
    def loss(self, x0_true, x_t, z, t):
        """Compute loss for training"""
        pass


class NBPosterior(BaseCountModel):
    """
    Negative Binomial approach: f_θ(x_t, t, z) → {r > 0, p ∈ (0,1)} 
    """
    
    @property
    def output_dim(self):
        return 2 * self.x_dim  # (log_r, logit_p)
    
    def process_output(self, h):
        log_r, logit_p = h.chunk(2, dim=-1)
        r = F.softplus(log_r) + 1e-3                  # keep strictly > 0
        p = torch.sigmoid(logit_p).clamp(1e-4, 1-1e-4)
        return r, p
    
    def sample(self, x_t, z, t, use_mean=False):
        """Sample from predicted NB distribution"""
        r, p = self.forward(x_t, z, t)
        dist = NegativeBinomial(total_count=r, probs=p)
        if use_mean:
            return dist.mean.round().long()
        else:
            return dist.sample().long()
    
    def loss(self, x0_true, x_t, z, t):
        """Negative log-likelihood under predicted NB"""
        r, p = self.forward(x_t, z, t)
        dist = NegativeBinomial(total_count=r, probs=p)
        return -(dist.log_prob(x0_true.float())).sum(-1).mean()


class BetaBinomialPosterior(BaseCountModel):
    """
    Beta-Binomial approach: f_θ(x_t, t, z) → {n ≥ 1, mean_p ∈ (0,1), concentration > 0}
    More stable parameterization with fixes for under-estimation
    """
    
    @property
    def output_dim(self):
        return 2 * self.x_dim  # (log_n, logit_mean_p, log_concentration)
    
    def process_output(self, h):
        logit_mean_p, log_concentration = h.chunk(2, dim=-1)
        
        # Force n to be much larger to avoid artificial ceiling
        mean_p = torch.sigmoid(logit_mean_p).clamp(0.0001, 0.9999) # Keep away from extremes
        concentration = F.softplus(log_concentration) 
        
        # Convert to alpha, beta with minimum values
        alpha = mean_p * concentration                    
        beta = (1 - mean_p) * concentration  
        
        return 100, alpha, beta
    
    def sample(self, x_t, z, t, use_mean=False):
        """Sample from predicted Beta-Binomial distribution"""
        n, alpha, beta = self.forward(x_t, z, t)
        # Convert to integers for total_count, but keep it large
        # n_int = n.round().long().clamp(min=10, max=500)  # Reasonable large bounds
        dist = BetaBinomial(total_count=n, concentration1=alpha, concentration0=beta)
        if use_mean:
            return dist.mean.round().long()
        else:
            return dist.sample().long()
    
    def loss(self, x0_true, x_t, z, t):
        """Negative log-likelihood under predicted Beta-Binomial with regularization"""
        n, alpha, beta = self.forward(x_t, z, t)
    
        
        # Also add penalty if mean_p * n is too different from observed counts
        # predicted_mean = n * alpha / (alpha + beta)
        # mean_penalty = F.mse_loss(predicted_mean, x0_true.float()) * 0.5
        
        # Convert to integers for likelihood calculation
        # n_int = n.round().long().clamp(min=int(n_min_needed), max=500)
        
        # Don't clamp x0_true since we ensure n is large enough
        dist = BetaBinomial(total_count=n, concentration1=alpha, concentration0=beta)
        nll = -(dist.log_prob(x0_true.float())).sum(-1).mean()
        
        return nll # + n_penalty + mean_penalty


class MLERegressor(BaseCountModel):
    """
    MLE approach: f_θ(x_t, t, z) → log(1 + x0_hat)
    """
    
    @property
    def output_dim(self):
        return self.x_dim  # log(1 + x0_hat)
    
    def process_output(self, h):
        return h  # raw log(1 + x0_hat) values
    
    def sample(self, x_t, z, t, use_mean=False):
        """Round predicted log values to get integer counts"""
        x0_hat = self.forward(x_t, z, t)
        return x0_hat.round().long()
    
    def loss(self, x0_true, x_t, z, t):
        """MSE loss on log(1 + x) scale"""
        x0_hat = self.forward(x_t, z, t)
        return F.mse_loss(x0_hat, x0_true.float())


class ZeroInflatedPoissonPosterior(BaseCountModel):
    """
    Zero-Inflated Poisson approach: f_θ(x_t, t, z) → {λ > 0, π ∈ (0,1)}
    More appropriate for count data than beta-binomial
    """
    
    @property
    def output_dim(self):
        return 2 * self.x_dim  # (log_lambda, logit_pi)
    
    def process_output(self, h):
        log_lambda, logit_pi = h.chunk(2, dim=-1)
        lam = F.softplus(log_lambda) + 1e-3           # Rate parameter > 0
        pi = torch.sigmoid(logit_pi).clamp(1e-4, 1-1e-4)  # Zero inflation probability
        return lam, pi
    
    def sample(self, x_t, z, t, use_mean=False):
        """Sample from predicted Zero-Inflated Poisson distribution"""
        lam, pi = self.forward(x_t, z, t)
        
        if use_mean:
            # Mean of ZIP: (1 - π) * λ  
            mean_val = (1 - pi) * lam
            return mean_val.round().long()
        else:
            # Sample from ZIP: with probability π return 0, else sample from Poisson(λ)
            poisson_samples = torch.distributions.Poisson(lam).sample()
            zero_mask = torch.bernoulli(pi).bool()
            zip_samples = torch.where(zero_mask, torch.zeros_like(poisson_samples), poisson_samples)
            return zip_samples.long()
    
    def loss(self, x0_true, x_t, z, t):
        """Negative log-likelihood under predicted ZIP"""
        lam, pi = self.forward(x_t, z, t)
        
        # ZIP log-likelihood
        # P(X=0) = π + (1-π)e^(-λ)
        # P(X=k) = (1-π) * λ^k * e^(-λ) / k!  for k > 0
        
        zero_mask = (x0_true == 0).float()
        nonzero_mask = 1 - zero_mask
        
        # Log-likelihood for zeros
        log_prob_zero = torch.log(pi + (1 - pi) * torch.exp(-lam))
        
        # Log-likelihood for non-zeros  
        poisson_dist = torch.distributions.Poisson(lam)
        log_prob_nonzero = torch.log(1 - pi) + poisson_dist.log_prob(x0_true.float())
        
        # Combine
        log_prob = zero_mask * log_prob_zero + nonzero_mask * log_prob_nonzero
        
        return -log_prob.sum(-1).mean() 


class IQNPosterior(BaseCountModel):
    """
    Implicit Quantile Networks approach: f_θ(x_t, t, z, τ) → quantile values
    Learns to predict quantiles of the count distribution directly
    """
    
    def __init__(self, x_dim, context_dim, hidden=128, n_quantiles=32, quantile_embedding_dim=64):
        # Initialize the base class first, but we'll override the network
        super().__init__(x_dim, context_dim, hidden)
        
        self.n_quantiles = n_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        
        # Override the network to handle quantile embeddings
        self.feature_net = nn.Sequential(
            nn.Linear(x_dim + 1 + context_dim, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
        )
        
        # Quantile embedding network (cosine embedding like in IQN paper)
        self.quantile_embedding = nn.Linear(quantile_embedding_dim, hidden)
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, self.x_dim)
        )
    
    @property
    def output_dim(self):
        return self.x_dim  # Direct quantile values
    
    def _get_quantile_embedding(self, tau):
        """
        Generate cosine embeddings for quantile levels τ
        tau: [batch_size, n_quantiles] or [batch_size, 1]
        """
        batch_size = tau.shape[0]
        n_tau = tau.shape[1]
        
        # Cosine embedding as in IQN paper
        i = torch.arange(1, self.quantile_embedding_dim + 1, 
                        dtype=torch.float32, device=tau.device).unsqueeze(0).unsqueeze(0)
        # Shape: [1, 1, quantile_embedding_dim]
        
        tau_expanded = tau.unsqueeze(-1)  # [batch_size, n_tau, 1]
        
        # Compute cosine embeddings
        embeddings = torch.cos(math.pi * i * tau_expanded)  # [batch_size, n_tau, embedding_dim]
        
        return self.quantile_embedding(embeddings)  # [batch_size, n_tau, hidden]
    
    def forward_with_quantiles(self, x_t, z, t, tau):
        """
        Forward pass with explicit quantile levels
        tau: [batch_size, n_quantiles] quantile levels in [0, 1]
        """
        # Get basic features
        t = t.expand_as(x_t[:, :1])
        inp = torch.cat([x_t.float(), t.float(), z.float()], dim=-1)
        features = self.feature_net(inp)  # [batch_size, hidden]
        
        # Get quantile embeddings
        quantile_emb = self._get_quantile_embedding(tau)  # [batch_size, n_quantiles, hidden]
        
        # Combine features with quantile embeddings
        batch_size, n_tau = tau.shape
        features_expanded = features.unsqueeze(1).expand(-1, n_tau, -1)  # [batch_size, n_quantiles, hidden]
        
        # Element-wise multiplication (Hadamard product)
        combined = features_expanded * quantile_emb  # [batch_size, n_quantiles, hidden]
        
        # Get quantile predictions
        quantiles = self.output_layer(combined)  # [batch_size, n_quantiles, x_dim]

        x_t_expanded = x_t.unsqueeze(1).expand(-1, n_tau, -1)
        
        return quantiles + x_t_expanded
    
    def process_output(self, h):
        # This method is required by base class but not used in IQN
        return h
    
    def forward(self, x_t, z, t):
        """
        Standard forward pass - sample random quantiles for training
        """
        batch_size = x_t.shape[0]
        
        # Sample random quantile levels
        tau = torch.rand(batch_size, self.n_quantiles, device=x_t.device)
        
        return self.forward_with_quantiles(x_t, z, t, tau)
    
    def sample(self, x_t, z, t, use_mean=False, n_samples=1):
        """
        Sample from predicted quantile distribution
        """
        batch_size = x_t.shape[0]
        
        if use_mean:
            # Use median (0.5 quantile) as point estimate
            tau = torch.full((batch_size, 1), 0.5, device=x_t.device)
            quantiles = self.forward_with_quantiles(x_t, z, t, tau)
            return quantiles.squeeze(1).round().long()
        else:
            # Sample random quantiles and interpolate
            tau = torch.rand(batch_size, n_samples, device=x_t.device)
            quantiles = self.forward_with_quantiles(x_t, z, t, tau)
            
            # For multiple samples, take one random sample per batch element
            if n_samples > 1:
                sample_idx = torch.randint(0, n_samples, (batch_size,), device=x_t.device)
                quantiles = quantiles[torch.arange(batch_size), sample_idx]
            else:
                quantiles = quantiles.squeeze(1)
                
            return quantiles.round().long()
    
    def loss(self, x0_true, x_t, z, t):
        """
        Quantile regression loss (Huber loss weighted by quantile levels)
        """
        batch_size = x_t.shape[0]
        
        # Sample random quantiles for training
        tau = torch.rand(batch_size, self.n_quantiles, device=x_t.device)
        
        # Get quantile predictions
        quantiles = self.forward_with_quantiles(x_t, z, t, tau)  # [batch_size, n_quantiles, x_dim]
        
        # Expand true values for comparison
        x0_expanded = x0_true.float().unsqueeze(1).expand(-1, self.n_quantiles, -1)  # [batch_size, n_quantiles, x_dim]
        tau_expanded = tau.unsqueeze(-1).expand(-1, -1, self.x_dim)  # [batch_size, n_quantiles, x_dim]
        
        # Compute quantile regression loss
        errors = x0_expanded - quantiles  # [batch_size, n_quantiles, x_dim]
        
        # Quantile loss: τ * max(errors, 0) + (1-τ) * max(-errors, 0)
        loss_positive = tau_expanded * F.relu(errors)
        loss_negative = (1 - tau_expanded) * F.relu(-errors)
        quantile_loss = loss_positive + loss_negative
        
        # Average over quantiles and sum over dimensions, then mean over batch
        return quantile_loss.mean(dim=1).sum(dim=1).mean() 


import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MMDPosterior(BaseCountModel):
    """
    Distributional-diffusion energy score (eq.14) with m-sample approximation.
    f_θ(x_t, t, z, ε) → x₀̂ = x_t + Δx
    """
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        hidden: int = 128,
        noise_dim: int = None,
        sigma: float = 1.0,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
    ):
        super().__init__(x_dim, context_dim, hidden)
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
        self._in_dim = x_dim + 1 + context_dim + self.noise_dim

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

    @property
    def output_dim(self):
        return self.x_dim

    def process_output(self, h):
        return h

    def forward(self, x_t, z, t, noise=None):
        """
        x_t: [B, x_dim]
        z:   [B, context_dim]
        t:   [B] or [B,1]
        noise: [B, noise_dim] or None (will be sampled)
        → returns [B, x_dim]
        """
        B = x_t.shape[0]
        # 1) sample noise if needed
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=x_t.device)

        # 2) unify t to shape [B,1]
        if t.dim() == 1:
            t_col = t.unsqueeze(-1)
        elif t.dim() == 2 and t.shape[1] == 1:
            t_col = t
        else:
            raise ValueError(f"t must be [B] or [B,1], got {tuple(t.shape)}")

        # 3) concat everything
        inp = torch.cat([x_t.float(), t_col.float(), z.float(), noise], dim=-1)

        # 4) sanity check
        assert inp.shape[1] == self._in_dim, (
            f"got inp dim={inp.shape[1]}, expected {self._in_dim}"
        )

        # 5) predict Δx, add to x_t
        return self.net(inp) + x_t

    def sample(self, x_t, z, t, use_mean=False):
        x0_hat = self.forward(x_t, z, t)
        return x0_hat.round().long()

    def _pairwise_dist(self, a, b, eps=1e-6):
        """
        a: [n, d], b: [m, d] → [n, m] of √(||a_i - b_j||² + eps)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)      # [n, m, d]
        sq   = (diff * diff).sum(-1)                # [n, m]
        return torch.sqrt(torch.clamp(sq, min=eps))

    def loss(self, x0_true, x_t, z, t):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||x0_trueᵢ - x̂ᵢⱼ||
            - (λ/(2(m-1))) ∑_{j≠j'} ||x̂ᵢⱼ - x̂ᵢⱼ'||
          ]
        """
        n, m, λ = x0_true.size(0), self.m, self.lambda_energy

        # — replicate each input m times —
        x_t_rep = x_t.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        z_rep   =   z.unsqueeze(1).expand(-1, m, -1).reshape(n * m, -1)
        # flatten t to 1D so forward() will turn it into [n*m,1]
        if t.dim() == 2 and t.shape[1] == 1:
            t_rep = t.expand(-1, m).reshape(n * m)
        elif t.dim() == 1:
            t_rep = t.unsqueeze(1).expand(-1, m).reshape(n * m)
        else:
            raise ValueError(f"t must be [n] or [n,1], got {tuple(t.shape)}")

        # sample m·n noises
        noise = torch.randn(n * m, self.noise_dim, device=x_t.device)

        # get all x̂ preds: [n*m, x_dim] → view [n, m, x_dim]
        x0_preds = self.forward(x_t_rep, z_rep, t_rep, noise)
        x0_preds = x0_preds.view(n, m, -1)  # [n, m, d]

        # 1) confinement
        x0_true_rep = x0_true.unsqueeze(1).expand(-1, m, -1)
        term_conf   = (x0_preds - x0_true_rep).norm(dim=2).mean(dim=1)  # [n]

        # 2) interaction (properly scaled!)
        pdists    = torch.stack([torch.pdist(x0_preds[i], p=2) for i in range(n)], dim=0)
        mean_pd   = pdists.mean(dim=1)                                # [n]
        term_int  = (λ / 2.0) * mean_pd                               # [n]

        return (term_conf - term_int).mean()

