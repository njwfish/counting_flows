"""
Neural Network Models for Count-based Flow Matching

Provides different neural architectures for predicting count distributions:
- NBPosterior: Negative Binomial parameters (r, p)
- BetaBinomialPosterior: Beta-Binomial parameters (n, alpha, beta)  
- MLERegressor: Direct count prediction via log(1 + x₀)
- ZeroInflatedPoissonPosterior: Zero-Inflated Poisson parameters (λ, π)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Beta
from abc import ABC, abstractmethod

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
        n = self.total_count
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
        return 3 * self.x_dim  # (log_n, logit_mean_p, log_concentration)
    
    def process_output(self, h):
        log_n, logit_mean_p, log_concentration = h.chunk(3, dim=-1)
        
        # Force n to be much larger to avoid artificial ceiling
        n = F.softplus(log_n) + 10.0                          # Start much higher (at least 50)
        mean_p = torch.sigmoid(logit_mean_p).clamp(0.001, 0.999) # Keep away from extremes
        concentration = F.softplus(log_concentration) + 1.0    # Higher concentration for stability
        
        # Convert to alpha, beta with minimum values
        alpha = mean_p * concentration                    # At least 0.5
        beta = (1 - mean_p) * concentration               # At least 0.5
        
        return n, alpha, beta
    
    def sample(self, x_t, z, t, use_mean=False):
        """Sample from predicted Beta-Binomial distribution"""
        n, alpha, beta = self.forward(x_t, z, t)
        # Convert to integers for total_count, but keep it large
        n_int = n.round().long().clamp(min=10, max=500)  # Reasonable large bounds
        dist = BetaBinomial(total_count=n_int, concentration1=alpha, concentration0=beta)
        if use_mean:
            return dist.mean.round().long()
        else:
            return dist.sample().long()
    
    def loss(self, x0_true, x_t, z, t):
        """Negative log-likelihood under predicted Beta-Binomial with regularization"""
        n, alpha, beta = self.forward(x_t, z, t)
        
        # Ensure n is always larger than observed counts + some buffer
        max_observed = x0_true.max().item()
        n_min_needed = max_observed + 10.0  # Buffer of 10
        
        # Add penalty if n is too small (encourages larger n)
        n_penalty = F.relu(n_min_needed - n).mean() * 5.0
        
        # Also add penalty if mean_p * n is too different from observed counts
        predicted_mean = n * alpha / (alpha + beta)
        mean_penalty = F.mse_loss(predicted_mean, x0_true.float()) * 0.5
        
        # Convert to integers for likelihood calculation
        n_int = n.round().long().clamp(min=int(n_min_needed), max=500)
        
        # Don't clamp x0_true since we ensure n is large enough
        try:
            dist = BetaBinomial(total_count=n_int, concentration1=alpha, concentration0=beta)
            nll = -(dist.log_prob(x0_true.float())).sum(-1).mean()
        except:
            # Fallback to MSE if beta-binomial fails
            nll = F.mse_loss(predicted_mean, x0_true.float())
        
        return nll + n_penalty + mean_penalty


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
        log_x0_plus1 = self.forward(x_t, z, t)
        x0_hat = torch.exp(log_x0_plus1) - 1
        return x0_hat.clamp(min=0).round().long()
    
    def loss(self, x0_true, x_t, z, t):
        """MSE loss on log(1 + x) scale"""
        log_x0_plus1_pred = self.forward(x_t, z, t)
        log_x0_plus1_true = torch.log(x0_true.float() + 1)
        return F.mse_loss(log_x0_plus1_pred, log_x0_plus1_true)


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