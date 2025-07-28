"""
Discrete Flow Bridge for Zero Masking

Implements proportional masking bridge for discrete flow matching.
At time t, each dimension is x_1 with probability t, otherwise x_0.
"""

import torch
import numpy as np
from typing import Dict, Any


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
    
    def __call__(self, x_0, x_1, t_target=None):
        """
        Apply zero masking bridge.
        
        Args:
            x_0: Source noise data
            x_1: Target data
            t_target: Optional target time (if None, sample random time)
            
        Returns:
            Dict with 'inputs' and 'output' for training
        """
        batch_size = x_0.shape[0]
        
        # Sample time
        if t_target is not None:
            t = torch.full((batch_size,), t_target, dtype=torch.float32, device=x_0.device)
        else:
            t = torch.rand(batch_size, device=x_0.device)
        
        # Apply zero masking: x_t = where(rand < t, x_1, x_0)
        # Each dimension independently chooses x_1 with probability t
        mask = torch.rand_like(x_0.float()) < t[:, None]
        x_t = torch.where(mask, x_0, x_1)
        
        return {
            "inputs": {
                "x_t": x_t.float(),  # Discrete masked state
                "t": t.unsqueeze(1),      # Time values (match CFM format)
            },
            "output": x_0.float(),   # Target for cross entropy loss
        }
    
    def sampler(
        self,
        x_1: torch.Tensor,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        **kwargs
    ):
        """
        Discrete time reverse sampler for Discrete Flow Matching.
        
        Args:
            x_1: End point samples (batch_size, dim) - start from here
            z: Additional conditioning dict
            model: Neural network model that predicts flow
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            
        Returns:
            x_0: Sampled starting points
            Optional: trajectory, x_hat predictions based on flags
        """
        batch_size, data_dim = x_1.shape
        vocab_size = 128  # Fixed vocab size
        
        # Forward sampling loop using exactly n_steps
        # Start from x_1 like other flow methods
        x_t = x_1.long()  # Start from x_1
        h = 1.0 / self.n_steps  # Fixed step size
        
        # Initialize tracking lists
        traj = [x_t.clone()] if return_trajectory else []
        xhat_traj = []
        
        for step in range(self.n_steps):
            t = step / self.n_steps  # Current time
            
            # Get model predictions (logits)
            t_tensor = torch.full((batch_size, 1), t, device=x_t.device)
            with torch.no_grad():
                logits = model.forward({"x_t": x_t, "t": t_tensor, **z})
                
            # Sample from predicted distribution
            p1 = torch.softmax(logits, dim=-1)
            
            one_hot_x_t = torch.nn.functional.one_hot(x_t, vocab_size).float()
            
            # Flow field: (p1 - one_hot_x_t) / (1.0 - t)
            if t < 1.0 - 1e-6:  # Avoid division by zero at t=1
                u = (p1 - one_hot_x_t) / (1.0 - t)
                
                # Update: one_hot_x_t + h * u
                new_probs = one_hot_x_t + h * u
                new_probs = torch.clamp(new_probs, min=0.0)
                new_probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-8)
                x_t = torch.distributions.Categorical(probs=new_probs).sample()
            else:
                # At final step, just sample from model prediction
                x_t = torch.distributions.Categorical(probs=p1).sample()
            
            if return_trajectory:
                traj.append(x_t.clone())
            if return_x_hat:
                # For discrete flow, sample from the predicted distribution for visualization
                sampled_pred = torch.distributions.Categorical(probs=p1).sample()
                xhat_traj.append(sampled_pred.clone())
        
        # Prepare outputs (match CFM format)
        outs = [x_t.float()]
        if return_trajectory:
            outs.append(torch.stack(traj).float())
        if return_x_hat:
            outs.append(torch.stack(xhat_traj).float())
        
        return tuple(outs) if len(outs) > 1 else x_t.float() 