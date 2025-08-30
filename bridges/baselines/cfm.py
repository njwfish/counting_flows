"""
Conditional Flow Matching Bridge

Clean implementation of CFM without external dependencies.
Implements the core CFM functionality directly in our framework.
"""

import torch
import numpy as np
import math
from typing import Dict, Any, Union, Tuple


def pad_t_like_x(t, x):
    """Pad time tensor to match the shape of x for broadcasting."""
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class CFMBridge:
    """
    Conditional Flow Matching Bridge.
    
    Implements the linear interpolation path: x_t = (1-t) * x_0 + t * x_1 + sigma * epsilon
    Flow field: u_t = x_1 - x_0
    """
    
    def __init__(self, n_steps: int = 100, fm_type: str = "cfm", ot_type: str = "exact", 
                 sigma: float = 1.0, device: int = 0):
        """
        Args:
            n_steps: Number of time steps for sampling
            fm_type: Flow matching type ("cfm", "variance_preserving", "schrodinger")
            ot_type: Optimal transport type ("exact" or "independent")  
            sigma: Noise level for the probability path
            device: Device for computations
        """
        self.n_steps = n_steps
        self.fm_type = fm_type
        self.ot_type = ot_type
        self.sigma = sigma
        self.device = device
        
        # Create discrete time points for reverse sampling
        self.time_points = torch.linspace(0, 1, n_steps + 1)

    def compute_mu_t(self, x_0, x_1, t):
        """
        Compute the mean of the probability path.
        Default: linear interpolation mu_t = (1-t) * x_0 + t * x_1
        """
        t_expanded = pad_t_like_x(t, x_0)
        return (1 - t_expanded) * x_0 + t_expanded * x_1

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path.
        Default: constant sigma
        """
        return self.sigma

    def compute_conditional_flow(self, x_0, x_1, t, x_t):
        """
        Compute the conditional vector field.
        Default: u_t = x_1 - x_0
        """
        return x_1 - x_0

    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                 t: Union[float, None] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward process: sample x_t and velocity field.
        
        Args:
            x_0: Source samples
            x_1: Target samples
            t: Optional target time (if None, sample random time)
            
        Returns:
            Tuple of (t, x_t, u_t) for training
                t: Time values (batch_size, 1)
                x_t: Interpolated samples at time t
                u_t: Velocity field (target for flow prediction)
        """
        batch_size = x_0.shape[0]
        x_0, x_1 = x_0.float(), x_1.float()
        
        # Sample time uniformly
        if t is not None:
            t = t.expand(batch_size, 1)
        else:
            if self.homogeneous_time:
                t = torch.rand(1, device=x_0.device).expand(batch_size, 1)
            else:
                t = torch.rand(batch_size, device=x_0.device)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Compute interpolation using overridable methods
        mu_t = self.compute_mu_t(x_0, x_1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x_0)
        x_t = mu_t + sigma_t * eps
        
        # Compute velocity field using overridable method
        u_t = self.compute_conditional_flow(x_0, x_1, t, x_t)
        
        return t.unsqueeze(1), x_t.float(), u_t.float()

    def sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
        use_sde: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Discrete time reverse sampler for Conditional Flow Matching.
        
        Args:
            x_1: End point samples (batch_size, dim)
            z: Additional conditioning dict
            model: Neural network model that predicts velocity field
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            guidance_x_0: Optional guidance target
            guidance_schedule: Optional guidance schedule
            use_sde: Whether to use SDE (Euler-Maruyama) instead of ODE
            **kwargs: Additional sampling arguments
        
        Returns:
            x_0: Sampled starting points
            Optional: trajectory, x_hat predictions based on flags
        """
        if use_sde:
            return self._sde_sampler(
                x_1, z, model, return_trajectory, return_x_hat, 
                guidance_x_0, guidance_schedule, **kwargs
            )
        else:
            return self._ode_sampler(
                x_1, z, model, return_trajectory, return_x_hat,
                guidance_x_0, guidance_schedule, **kwargs
            )

    def _ode_sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Discrete time reverse sampler for Conditional Flow Matching.
        
        Args:
            x_1: End point samples (batch_size, dim)
            z: Additional conditioning dict
            model: Neural network model that predicts velocity field
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            guidance_x_0: Optional guidance target
            guidance_schedule: Optional guidance schedule
            **kwargs: Additional sampling arguments
        
        Returns:
            x_0: Sampled starting points
            Optional: trajectory, x_hat predictions based on flags
        """
        b, d = x_1.shape
        x_t = x_1.float()
        
        # Move time points to same device as input
        time_points = self.time_points.to(x_t.device)
        
        if guidance_x_0 is not None:
            guidance_x_0 = guidance_x_0.float()
        
        def sample_step(k, x_t, **z):
            """Single reverse sampling step using discrete time"""
            # Current time point
            t_curr = time_points[k]
            t = t_curr.expand(b, 1)
            
            # Predict velocity field
            with torch.no_grad():
                v_pred = model.sample(x_t=x_t, t=t, **z)
                
            # Apply guidance if provided
            if guidance_x_0 is not None and guidance_schedule is not None:
                # Simple guidance: interpolate velocity towards guidance target
                guidance_weight = guidance_schedule[k] 
                x_0_pred = x_t - (1 - t_curr) * v_pred
                x_0_guided = guidance_weight * guidance_x_0 + (1 - guidance_weight) * x_0_pred
                v_pred = (x_t - x_0_guided) / (1 - t_curr + 1e-8)
            
            # Take discrete step backwards in time
            dt = time_points[k] - time_points[k-1]  # Positive step size  
            x_next = x_t - dt * v_pred  # Negative velocity for backward flow
            
            # Estimate x_0 prediction for this step
            x_0_pred = x_t - (1 - t_curr) * v_pred
            
            return x_next, x_0_pred
        
        # Initialize tracking lists
        traj = [x_t]
        xhat_traj = []
        
        # Reverse sampling loop
        for k in range(self.n_steps, 0, -1):
            x_t, x_0_pred = sample_step(k, x_t, **z)
            
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

    def _sde_sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        SDE sampler using Euler-Maruyama method.
        
        Implements: dx = f(x,t)dt + g(x,t)dW
        where f(x,t) = -v_t(x) (negative velocity for backward flow)
        and g(x,t) = sigma_t(t) (uses same noise schedule as forward process)
        """
        b, d = x_1.shape
        x_t = x_1.float()
        
        # Move time points to same device as input
        time_points = self.time_points.to(x_t.device)
        
        if guidance_x_0 is not None:
            guidance_x_0 = guidance_x_0.float()
        
        def sde_sample_step(k, x_t, **z):
            """Single SDE step using Euler-Maruyama"""
            # Current time point
            t_curr = time_points[k]
            t = t_curr.expand(b, 1)
            
            # Predict velocity field (drift)
            with torch.no_grad():
                v_pred = model.sample(x_t=x_t, t=t, **z)
                
            # Apply guidance if provided
            if guidance_x_0 is not None and guidance_schedule is not None:
                guidance_weight = guidance_schedule[k] 
                x_0_pred = x_t - (1 - t_curr) * v_pred
                x_0_guided = guidance_weight * guidance_x_0 + (1 - guidance_weight) * x_0_pred
                v_pred = (x_t - x_0_guided) / (1 - t_curr + 1e-8)
            
            # Euler-Maruyama step
            dt = time_points[k] - time_points[k-1]  # Positive step size
            
            # Drift term: f(x,t) = -v_t(x) for backward flow
            drift = -v_pred
            
            # Diffusion term: g(x,t) = sigma_t(t) - use same as forward process
            sigma_t = self.compute_sigma_t(t_curr)
            sigma_t = pad_t_like_x(sigma_t, x_t)
            noise = torch.randn_like(x_t)
            diffusion = sigma_t * noise
            
            # SDE step: x_next = x_t + f*dt + g*sqrt(dt)*dW
            x_next = x_t + drift * dt + diffusion * torch.sqrt(dt)
            
            # Estimate x_0 prediction for this step
            x_0_pred = x_t - (1 - t_curr) * v_pred
            
            return x_next, x_0_pred
        
        # Initialize tracking lists
        traj = [x_t]
        xhat_traj = []
        
        # Reverse SDE sampling loop
        for k in range(self.n_steps, 0, -1):
            x_t, x_0_pred = sde_sample_step(k, x_t, **z)
            
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
        return f'CFMBridge(n_steps={self.n_steps}, fm_type={self.fm_type}, ot_type={self.ot_type}, sigma={self.sigma})'

    def __repr__(self):
        return self.__str__()


class SchrodingerBridgeConditionalFlowMatcher(CFMBridge):
    """
    Schrödinger Bridge Conditional Flow Matcher.
    
    Implements time-dependent noise schedule: sigma_t = sigma * sqrt(t * (1 - t))
    And modified flow field: u_t = (1 - 2*t) / (2*t*(1-t)) * (x_t - mu_t) + x_1 - x_0
    """
    
    def compute_sigma_t(self, t):
        """
        Schrödinger bridge noise schedule: sigma * sqrt(t * (1 - t))
        """
        return self.sigma * torch.sqrt(t * (1 - t))
    
    def compute_conditional_flow(self, x_0, x_1, t, x_t):
        """
        Schrödinger bridge flow field:
        u_t = (1 - 2*t) / (2*t*(1-t)) * (x_t - mu_t) + x_1 - x_0
        """
        t_expanded = pad_t_like_x(t, x_0)
        mu_t = self.compute_mu_t(x_0, x_1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t_expanded) / (2 * t_expanded * (1 - t_expanded) + 1e-8)
        u_t = sigma_t_prime_over_sigma_t * (x_t - mu_t) + x_1 - x_0
        return u_t


class VariancePreservingConditionalFlowMatcher(CFMBridge):
    """
    Variance Preserving Conditional Flow Matcher with trigonometric interpolants.
    
    Uses cosine/sine interpolation: mu_t = cos(π*t/2)*x_0 + sin(π*t/2)*x_1
    Flow field: u_t = π/2 * (cos(π*t/2)*x_1 - sin(π*t/2)*x_0)
    """
    
    def compute_mu_t(self, x_0, x_1, t):
        """
        Trigonometric interpolation: cos(π*t/2)*x_0 + sin(π*t/2)*x_1
        """
        t_expanded = pad_t_like_x(t, x_0)
        return torch.cos(math.pi / 2 * t_expanded) * x_0 + torch.sin(math.pi / 2 * t_expanded) * x_1
    
    def compute_conditional_flow(self, x_0, x_1, t, x_t):
        """
        Trigonometric flow field: π/2 * (cos(π*t/2)*x_1 - sin(π*t/2)*x_0)
        """
        t_expanded = pad_t_like_x(t, x_0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t_expanded) * x_1 - torch.sin(math.pi / 2 * t_expanded) * x_0)