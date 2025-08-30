"""
Discrete Flow Bridge for Zero Masking

Implements proportional masking bridge for discrete flow matching.
At time t, each dimension is x_1 with probability t, otherwise x_0.
"""

import torch
import numpy as np
from typing import Dict, Any, Union, Tuple


class DiscreteFlowBridge:
    """
    Zero masking bridge for discrete flow matching.
    
    Implements: x_t = where(rand < t, x_1, x_0)
    Each dimension independently chooses x_1 with probability t.
    """
    
    def __init__(self, n_steps: int = 100, device: int = 0):
        """
        Args:
            n_steps: Number of time steps for sampling
            device: Device for computations
        """
        self.n_steps = n_steps
        self.device = device
        
        # Create discrete time points for reverse sampling
        self.time_points = torch.linspace(0, 1, n_steps + 1)
    
    def __call__(self, x_0, x_1, t=None):
        """
        Apply zero masking bridge.
        
        Args:
            x_0: Source noise data
            x_1: Target data
            t: Optional target time (if None, sample random time)
            
        Returns:
            Dict with 'inputs' and 'output' for training
        """
        batch_size = x_0.shape[0]
        
        # Sample time
        if t is not None:
            t = t.expand(batch_size, 1)
        else:
            if self.homogeneous_time:
                t = torch.rand(1, device=x_0.device).expand(batch_size, 1)
            else:
                t = torch.rand(batch_size, device=x_0.device)
        
        # Apply zero masking: x_t = where(rand < t, x_1, x_0)
        # Each dimension independently chooses x_1 with probability t
        mask = torch.rand_like(x_0.float()) < t[:, None]
        x_t = torch.where(mask, x_0, x_1)
        
        return t.unsqueeze(1), x_t.float(), x_0.float()
    
    def sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Discrete time reverse sampler for Discrete Flow Matching.
        
        Args:
            x_1: End point samples (batch_size, dim) - start from here
            z: Additional conditioning dict
            model: Neural network model that predicts flow
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            **kwargs: Additional sampling arguments
            
        Returns:
            x_0: Sampled starting points
            Optional: trajectory, x_hat predictions based on flags
        """
        b, d = x_1.shape
        x_t = x_1.long()
        
        # Move time points to same device as input
        time_points = self.time_points.to(x_t.device)
        
        def sample_step(k, x_t, **z):
            """Single forward sampling step using discrete time"""
            # Current time point (note: DFM goes forward from 0 to 1)
            t_curr = time_points[k-1]  # k-1 because we're going forward
            t = t_curr.expand(b, 1)
            
            # Get model predictions (logits)
            with torch.no_grad():
                logits = model.forward({"x_t": x_t.float(), "t": t, **z})
            
            # Sample from predicted distribution
            p1 = torch.softmax(logits, dim=-1)
            
            # For discrete flow, the x_0 prediction is just sampling from p1
            x_0_pred = torch.distributions.Categorical(probs=p1).sample()
            
            one_hot_x_t = torch.nn.functional.one_hot(x_t, p1.shape[-1]).float()
            
            # Flow field: (p1 - one_hot_x_t) / (1.0 - t)
            if t_curr < 1.0 - 1e-6:  # Avoid division by zero at t=1
                u = (p1 - one_hot_x_t) / (1.0 - t_curr)
                
                # Take discrete step forward in time
                dt = time_points[k] - time_points[k-1]
                new_probs = one_hot_x_t + dt * u
                new_probs = torch.clamp(new_probs, min=0.0)
                new_probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-8)
                x_next = torch.distributions.Categorical(probs=new_probs).sample()
            else:
                # At final step, just sample from model prediction
                x_next = x_0_pred
            
            return x_next, x_0_pred
        
        # Initialize tracking lists
        traj = [x_t]
        xhat_traj = []
        
        # Forward sampling loop (note: DFM goes forward, unlike CFM/Diffusion)
        for k in range(1, self.n_steps + 1):
            x_t, x_0_pred = sample_step(k, x_t, **z)
            
            if return_trajectory:
                traj.append(x_t)
            if return_x_hat:
                xhat_traj.append(x_0_pred)
        
        # Prepare outputs (match CFM format)
        outs = [x_t.float()]
        if return_trajectory:
            outs.append(torch.stack(traj).float())
        if return_x_hat:
            outs.append(torch.stack(xhat_traj).float())
        
        return tuple(outs) if len(outs) > 1 else x_t.float() 