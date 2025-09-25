import torch
import numpy as np
import math
from typing import Dict, Any, Union, Tuple



class NoBridge:
    """
    Conditional Flow Matching Bridge.
    
    Implements the linear interpolation path: x_t = (1-t) * x_0 + t * x_1 + sigma * epsilon
    Flow field: u_t = x_1 - x_0
    """
    
    def __init__(self):
        """
        Args:
            sigma: Noise level for the probability path
            device: Device for computations
        """

    def __call__(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                 t: Union[float, None] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        No bridge, just return the target
        """
        batch_size = x_0.shape[0]
        x_0, x_1 = x_0.float(), x_1.float()
        
        t = torch.ones(batch_size, 1, device=x_0.device)
        
        return t.unsqueeze(1), torch.zeros_like(x_1.float()), x_0.float()

    def sampler(
        self,
        x_1: torch.Tensor,
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
        n_steps: int = 10,
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
            n_steps: Number of time steps for sampling
            use_sde: Whether to use SDE (Euler-Maruyama) instead of ODE
            **kwargs: Additional sampling arguments
        
        Returns:
            x_0: Sampled starting points
            Optional: trajectory, x_hat predictions based on flags
        """
        b = x_1.shape[0]
        x_t = x_1.float()
        
        if guidance_x_0 is not None:
            guidance_x_0 = guidance_x_0.float()

        # Initialize tracking lists
        traj = [x_t]
        xhat_traj = []
        
        # Reverse sampling loop
        t = torch.ones(b, 1, device=x_t.device)
        x_0_pred = model.sample(x_t=x_t, t=t, **z)
        
            
        if return_trajectory:
            traj.append(x_0_pred)
        if return_x_hat:
            xhat_traj.append(x_0_pred)
        
        # Prepare outputs
        outs = [x_t]
        if return_trajectory:
            outs.append(torch.stack(traj))
        if return_x_hat:
            outs.append(torch.stack(xhat_traj))
        
        return tuple(outs) if len(outs) > 1 else x_t
