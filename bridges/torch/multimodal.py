import numpy as np
import torch
from typing import Dict, Any, Union, Tuple


class MultimodalBridge:
    def __init__(self, bridges):
        """
        Define a bridge that combines multiple bridges.

        Args:
            bridges: Dict of bridges where the key is the name and the value is a bridge.
        """
        super().__init__()
        self.bridges = bridges
        self.keys = list(bridges.keys())
    
    def __call__(self, x_0, x_1, t=None):
        # Use same time for all modalities

        # batch size
        base_arr = x_0[self.keys[0]]
        b = base_arr.shape[0]
        device = base_arr.device

        if t is None:            
            t = torch.rand(b, 1, device=device)

        x_t = {}
        target = {}
        
        for k in x_0:
            t_k, x_t_k, target_k = self.bridges[k](x_0[k], x_1[k], t=t)
            x_t[k] = x_t_k
            target[k] = target_k
            # Use time from first modality (should be same for all)
            if k == list(x_0.keys())[0]:
                time_output = t_k
                
        return time_output, x_t, target

    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        x_t_next, x_0_next = {}, {}
        for k in x_t:  # Use x_t keys instead of x_0
            x_t_next[k], x_0_next[k] = self.bridges[k].sample_step(t_curr, t_next, x_t[k], x_0_pred[k], **z)
        return x_t_next, x_0_next


    def sampler(
        self,
        x_1: Dict[str, Any],
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        n_steps: int = 10,
        **kwargs
    ) -> Union[Any, Tuple[Any, ...]]:
        """
        Discrete time reverse sampler for Multimodal Bridges.
        
        Args:
            x_1: End point samples (batch_size, dim) - starting noise
            z: Additional conditioning dict
            model: Neural network model that predicts x_0
            return_trajectory: Whether to return full sampling trajectory
            return_x_hat: Whether to return predicted x_0 at each step
            n_steps: Number of time steps for sampling
            **kwargs: Additional sampling arguments
            
        Returns:
            x_0: Final samples
            Optional: trajectory, x_hat predictions based on flags
        """
        # Get batch size from any modality
        b = next(iter(x_1.values())).shape[0]
        x_t = {k: v.float() for k, v in x_1.items()}
        
        # Move time points to same device as input
        device = next(iter(x_t.values())).device
        time_points = torch.linspace(0, 1, n_steps + 1).to(device)
        
        # Initialize tracking lists
        traj = {k: [x_t[k]] for k in x_t}
        xhat_traj = {k: [] for k in x_t}
        
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
                for k in x_t:
                    traj[k].append(x_t[k])
            if return_x_hat:
                for k in x_0_pred:
                    xhat_traj[k].append(x_0_pred[k])
        
        # Prepare outputs
        outs = [x_t]
        if return_trajectory:
            for k in x_t:
                traj[k] = torch.stack(traj[k])
            outs.append(traj)
        if return_x_hat:
            for k in x_t:
                xhat_traj[k] = torch.stack(xhat_traj[k])
            outs.append(xhat_traj)
        
        return tuple(outs) if len(outs) > 1 else x_t
