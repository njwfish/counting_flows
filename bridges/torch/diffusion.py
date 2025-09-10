"""
VPSDE Diffusion Bridge Implementation

A bridge implementation for Variance Preserving Stochastic Differential Equations (VPSDE)
that follows the framework pattern. This implementation focuses on x_0 prediction and 
uses the VPSDE formulation from the provided diffusion_uvit.py.

Since this is not a traditional bridge, it samples x_1 from Gaussian noise and
generates x_t via the forward diffusion process defined by the VPSDE.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, Union


def stp(s, ts: torch.Tensor):
    """Scalar tensor product - broadcast scalar s to match tensor ts dimensions"""
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def duplicate(tensor, *size):
    """Duplicate tensor across batch dimension"""
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class VPSDE:
    """
    Variance Preserving Stochastic Differential Equation
    dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
    """
    
    def __init__(self, beta_min=0.1, beta_max=20):
        """
        Args:
            beta_min: Minimum noise schedule value
            beta_max: Maximum noise schedule value
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        """Drift coefficient f(x, t)"""
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        """Diffusion coefficient g(t)"""
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):
        """Beta(t) - noise schedule"""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):
        """Integral of beta(tau) from s to t"""
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_alpha(self, s, t):
        """Alpha coefficient for skip connection from time s to t"""
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def skip_beta(self, s, t):
        """Beta coefficient for skip connection from time s to t"""
        return 1. - self.skip_alpha(s, t)

    def cum_alpha(self, t):
        """Cumulative alpha from time 0 to t"""
        return self.skip_alpha(0, t)

    def cum_beta(self, t):
        """Cumulative beta from time 0 to t"""
        return self.skip_beta(0, t)

    def snr(self, t):
        """Signal-to-noise ratio"""
        return 1. / self.nsr(t)

    def nsr(self, t):
        """Noise-to-signal ratio"""
        return self.squared_diffusion_integral(0, t).expm1()

    def marginal_prob(self, x0, t):
        """Mean and std of q(xt|x0)"""
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample_forward(self, x0, eps, t):
        """Sample from forward diffusion process q(xt|x0)"""
        mean, std = self.marginal_prob(x0, t)
        xt = mean + stp(std, eps)
        return xt


class DiffusionBridge:
    """
    VPSDE Diffusion Bridge for the framework
    
    This is not a traditional bridge but implements the interface:
    - __call__: Forward diffusion process (x_0 -> x_t)  
    - sampler: Reverse diffusion process (x_1/noise -> x_0)
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, device: int = 0, homogeneous_time: bool = False):
        """
        Args:
            beta_min: Minimum noise schedule value
            beta_max: Maximum noise schedule value  
            device: Device for computations
        """
        self.device = device
        self.homogeneous_time = homogeneous_time
        self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max)

    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                 t: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample x_t from q(x_t|x_0)
        
        Args:
            x_0: Clean data samples
            x_1: Must be gaussian noise 
            t: Target time (if None, sample random time)
            
        Returns:
            Tuple of (t, x_t, x_0) for training
                t: Time values (batch_size, 1)
                x_t: Noisy samples at time t
                x_0: Original clean data (target for x_0 prediction)
        """
        batch_size = x_0.shape[0]
        x_0 = x_0.float()
        
        # Sample time uniformly

        if t is not None:
            if isinstance(t, float):
                t = torch.full((batch_size,), t, device=x_0.device)
            else:
                t = t.to(x_0.device)
        else:
            if self.homogeneous_time:
                t = torch.rand(1, device=x_0.device).expand(batch_size, 1)
            else:
                t = torch.rand(batch_size, device=x_0.device)
        
        # Sample x_t from forward process q(x_t|x_0)
        x_t = self.sde.sample_forward(x_0, x_1, t)
        
        # Return (t, x_t, target) where target is x_0 for x_0 prediction
        return t.unsqueeze(1), x_t.float(), x_0.float()


    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        """Single reverse sampling step using discrete time"""
        if t_curr == 0.0:
            return x_0_pred, x_0_pred
            
        if isinstance(t_curr, float):
            t_curr, t_next = torch.tensor(t_curr, device=x_t.device), torch.tensor(t_next, device=x_t.device)
        # Compute reverse SDE step from x_0 prediction
        alpha = self.sde.cum_alpha(t_curr)
        beta = self.sde.cum_beta(t_curr)
        
        # Convert x_0 prediction to noise prediction
        noise_pred = (x_t - stp(alpha ** 0.5, x_0_pred)) / stp(beta ** 0.5, torch.ones_like(x_t))
        
        # Compute score function: score = -noise / sqrt(beta)
        score = stp(-beta.rsqrt(), noise_pred)
        
        # Reverse drift: f(x,t) - g(t)^2 * score(x,t)
        drift = self.sde.drift(x_t, t_curr)
        diffusion = self.sde.diffusion(t_curr)
        reverse_drift = drift - stp(diffusion ** 2, score)
        
        # Take discrete step backwards in time (Euler-Maruyama)
        dt = t_curr - t_next
        mean = x_t - reverse_drift * dt
        sigma = diffusion * dt.sqrt()
        x_next = mean + stp(sigma, torch.randn_like(x_t)) if t_next != 0 else mean
        
        return x_next, x_0_pred

    def sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        n_steps: int = 10,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Discrete time reverse sampler for VPSDE Diffusion.
        
        Args:
            x_1: End point samples (batch_size, dim) - starting noise
            z: Additional conditioning dict
            model: Neural network model that predicts x_0
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            **kwargs: Additional sampling arguments
            
        Returns:
            x_0: Final samples
            Optional: trajectory, x_hat predictions based on flags
        """
        b = x_1.shape[0]
        x_t = x_1.float()
        
        # Move time points to same device as input
        time_points = torch.linspace(0, 1, n_steps + 1).to(x_t.device)
        
        # Initialize tracking lists
        traj = [x_t]
        xhat_traj = []
        
        # Reverse sampling loop
        for k in range(n_steps, 0, -1):
            # Current time point
            t_curr = time_points[k]
            t_next = time_points[k-1]
            t = t_curr.expand(b, 1)
            
            # Predict x_0 from current noisy state
            with torch.no_grad():
                x_0_pred = model.sample(x_t=x_t, t=t, **z)
                x_t, x_0_pred = self.sample_step(t_curr, t_next, x_t, x_0_pred, **z)
            
            if return_trajectory:
                traj.append(x_t)
            if return_x_hat:
                xhat_traj.append(x_0_pred)
        
        # Prepare outputs
        outs = [x_t]
        if return_trajectory:
            outs.append(torch.stack(traj))
        if return_x_hat:
            outs.append(torch.stack(xhat_traj))
        
        return tuple(outs) if len(outs) > 1 else x_t

    def __str__(self):
        return f'DiffusionBridge(beta_min={self.sde.beta_0}, beta_max={self.sde.beta_1})'

    def __repr__(self):
        return self.__str__()
