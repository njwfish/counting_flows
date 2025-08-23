import torch
import torch.nn as nn

class DeconvolutionEnergyScoreLoss(nn.Module):
    """
    Distributional-diffusion energy score loss (eq.14) with m-sample approximation.
    Works with arbitrary architectures and clean input interface.
    """
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
        min_value: int = 0,
        value_range: int = 10,
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy
        self.min_value = min_value
        self.value_range = value_range
        if self.min_value == 0:
            self.act_fn = nn.Softplus()
        else:
            self.act_fn = nn.Identity()

    def _pairwise_dist(self, a, b, eps=1e-6):
        """
        Compute pairwise distances for energy score
        a: [n, d], b: [m, d] → [n, m] of √(||a_i - b_j||² + eps)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)      # [n, m, d]
        sq   = (diff * diff).sum(-1)                # [n, m]
        return torch.sqrt(torch.clamp(sq, min=eps))

    def forward(self, inputs):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            
        Returns:
            Prediction tensor
        """
        # Add noise (energy score specific requirement)
        inputs_with_noise = inputs.copy()
        base_input = list(inputs.values())[0]
        batch_size = base_input.shape[0]
        inputs_with_noise['noise'] = torch.randn(batch_size, self.noise_dim, device=base_input.device)
        
        return self.act_fn(self.architecture(**inputs_with_noise))

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        """
        prediction = self.forward(kwargs)
        return prediction.round().long()

    def loss(self, target, inputs):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||target_i - pred_ij||
            - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
          ]
        """
        base_input = list(inputs.values())[0]
        n, λ = base_input.shape[0], self.lambda_energy
        num_groups = target.shape[0]
        group_size = n // num_groups

        # Replicate all inputs m times by iterating over the dict
        replicated_inputs = {}
        for key, value in inputs.items():
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.architecture(**replicated_inputs)
        predictions = predictions.view(num_groups, group_size, self.m, -1)  # [n, num_groups, m, d]
        predictions = predictions.sum(dim=1)  # [n, m, d]

        # Confinement term: distance to target
        target_expanded = target.unsqueeze(1).expand(-1, self.m, -1)
        term_conf = (predictions - target_expanded).norm(dim=2).mean(dim=1)  # [n]

        # Interaction term (efficient batched computation)
        # Using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ identity
        sq = predictions.pow(2).sum(dim=2)  # [n, m] - squared norms
        inn = torch.bmm(predictions, predictions.transpose(1,2))  # [n, m, m] - inner products
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2*inn  # [n, m, m] - squared distances
        sqd = torch.clamp(sqd, min=1e-6)  # avoid sqrt(0)
        d = sqd.sqrt()  # [n, m, m] - distances
        
        # Mean of off-diagonal pairwise distances
        # Create mask for off-diagonal elements on the fly
        m_mask = torch.ones(self.m, self.m, device=predictions.device) - torch.eye(self.m, device=predictions.device)
        mean_pd = (d * m_mask).sum(dim=(1,2)) / (self.m * (self.m - 1))  # [n]
        term_int = (λ / 2.0) * mean_pd  # [n]

        return (term_conf - term_int).mean()



