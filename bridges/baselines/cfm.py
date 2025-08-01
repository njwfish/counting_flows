from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
import torch


class CFMBridge:
    def __init__(self, n_steps, fm_type="cfm", ot_type="exact", sigma=1.0, device=0):
        self.n_steps = n_steps
        self.fm_type = fm_type
        self.ot_type = ot_type
        self.device = device

        if self.fm_type == "cfm":
            if self.ot_type == "exact":
                fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            else:
                fm = ConditionalFlowMatcher(sigma=sigma)
        elif self.fm_type == "schrodinger":
            fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, flow_type=self.ot_type)
        elif self.fm_type == "variance_preserving":
            fm = VariancePreservingConditionalFlowMatcher(sigma=sigma, flow_type=self.ot_type)
        else:
            raise ValueError(f"Invalid flow matching type: {self.fm_type}")

        self.fm = fm
        
        # Create discrete time points for reverse sampling
        self.time_points = torch.linspace(0, 1, n_steps + 1)

    def __call__(self, x_0, x_1, t_target=None):
        x_0, x_1 = x_0.float(), x_1.float()
        if t_target is not None:
            t_target = torch.tensor(t_target).broadcast_to(x_0.shape[0], 1).to(x_0.device)
        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x_0, x_1, t=t_target)
        return {
            "inputs": {
                "t": t.unsqueeze(1),
                "x_t": x_t,
            },
            "output": u_t,
        }

    def sampler(
        self,
        x_1: torch.Tensor,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        guidance_x_0: torch.Tensor = None,
        guidance_schedule: torch.Tensor = None,
    ):
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
            dt = time_points[k] - time_points[k-1]
            x_next = x_t - dt * v_pred
            
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

