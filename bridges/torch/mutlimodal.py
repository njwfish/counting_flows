from torch import nn

class MultimodalBridge(nn.Module):
    def __init__(self, bridges):
        """
        Define a bridge that combines multiple bridges.

        Args:
            bridges: Dict of bridges where the key is the name and the value is a bridge.
        """
        super().__init__()
        self.bridges = bridges
    
    def __call__(self, x_0, x_1, t=None):
        if t is None:
            t = torch.rand()

        for k in x_0:
            t, x_t, x_0 = self.bridges[k](x_0[k], x_1[k], t=t)
        return t, x_t, x_0

    def sample_step(self, t_curr, t_next, x_t, x_0_pred, **z):
        x_t_next, x_0_next = {}, {}
        for k in x_0:
            x_t_next[k], x_0_next[k] = self.bridges[k].sample_step(t_curr, t_next, x_t[k], x_0_pred[k], **z)
        return x_t_next, x_0_next


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
        b = x_1.shape[0]
        x_t = x_1.float()
        
        # Move time points to same device as input
        time_points = torch.linspace(0, 1, n_steps + 1).to(x_t.device)
        
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
                for k in x_t:
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
