import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, Optional


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
            t_curr = time_points[k].item()
            t_next = time_points[k-1].item()
            t = time_points[k].expand(b, 1)
            
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


class MultiTimeDimensionalBridge(MultimodalBridge):
    def __init__(self, bridges):
        """
        Define a bridge that combines multiple bridges.

        Args:
            bridges: Dict of bridges where the key is the name and the value is a bridge.
        """
        super().__init__(bridges)
        self.time_dim = len(bridges)
    
    def __call__(self, x_0, x_1, t=None):
        # Use same time for all modalities

        # batch size
        base_arr = x_0[self.keys[0]]
        b = base_arr.shape[0]
        device = base_arr.device

        x_t = {}
        target = {}

        t_out = {}
        for bridge in x_0:
            t_k, x_t_k, target_k = self.bridges[bridge](x_0[bridge], x_1[bridge], t=t)
            t_out[bridge] = t_k
            x_t[bridge] = x_t_k
            target[bridge] = target_k
            
        return t_out, x_t, target

    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        x_t_next, x_0_next = {}, {}
        for bridge in x_t:  # Use x_t keys instead of x_0
            x_t_next[bridge], x_0_next[bridge] = self.bridges[bridge].sample_step(t_curr[bridge], t_next[bridge], x_t[bridge], x_0_pred[bridge], **z)
        return x_t_next, x_0_next


    def sampler(
        self,
        x_1: Dict[str, Any],
        z: Dict[str, Any],
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        n_steps: int = 10,
        start_times: Optional[Dict[str, Any]] = None,
        x_start_time: Optional[Dict[str, Any]] = None,
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
            start_times: Optional start times for each bridge, so we can diffuse one modality from time 1 but condition on the end time zero for another
            x_start_time: Optional start times for each bridge, so we can diffuse one modality from time 1 but condition on the end time zero for another
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
        time_points = {}
        if start_times is not None:
            for bridge in x_t:
                if bridge not in start_times:
                    start_times[bridge] = 1.0
                time_points[bridge] = torch.linspace(0.0, start_times[bridge], n_steps + 1).to(device)
                x_1[bridge] = x_start_time[bridge] if bridge in x_start_time else x_1[bridge]
        else:
            time_points = {bridge: torch.linspace(0, 1, n_steps + 1).to(device) for bridge in x_t}
        
        # Initialize tracking lists
        traj = {bridge: [x_t[bridge]] for bridge in x_t}
        xhat_traj = {bridge: [] for bridge in x_t}
        
        # Reverse sampling loop
        for k in range(n_steps, 0, -1):
            # Current time point
            t_curr = {bridge: time_points[bridge][k].item() for bridge in x_t}
            t_next = {bridge: time_points[bridge][k-1].item() for bridge in x_t}
            t = {bridge: time_points[bridge][k].expand(b, 1) for bridge in x_t}

            # Predict x_0 from current noisy state
            with torch.no_grad():
                x_0_pred = model.sample(x_t=x_t, t=t, **z)
                x_t, x_0_pred = self.sample_step(t_curr, t_next, x_t, x_0_pred, **z)
            
            if return_trajectory:
                for bridge in x_t:
                    traj[bridge].append(x_t[bridge])
            if return_x_hat:
                for bridge in x_0_pred:
                    xhat_traj[bridge].append(x_0_pred[bridge])
        
        # Prepare outputs
        outs = [x_t]
        if return_trajectory:
            for bridge in x_t:
                traj[bridge] = torch.stack(traj[bridge])
            outs.append(traj)
        if return_x_hat:
            for bridge in x_t:
                xhat_traj[bridge] = torch.stack(xhat_traj[bridge])
            outs.append(xhat_traj)
        
        return tuple(outs) if len(outs) > 1 else x_t
