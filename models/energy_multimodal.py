import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalEnergyScoreLoss(nn.Module):
    """
    Multimodal energy score loss that computes energy scores for both images and counts.
    
    This is the basic version without deconvolution aggregation - it treats each modality
    independently and combines their energy scores.
    
    Energy score formula (per modality):
    L = mean_i [
        (1/m) ∑_j ||target_i - pred_ij||
        - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
    ]
    
    Total loss is weighted combination of image and count energy scores.
    """
    def __init__(
        self,
        architecture,
        noise_dim: int = 16,
        m_samples: int = 16,
        lambda_energy: float = 1.0,
        img_weight: float = 1.0,
        count_weight: float = 1.0,
        min_count_value: int = 0,
        count_value_range: int = 10,
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy
        self.img_weight = img_weight
        self.count_weight = count_weight
        self.min_count_value = min_count_value
        self.count_value_range = count_value_range
        
        # Activation function for counts (to ensure non-negativity)
        if self.min_count_value == 0:
            self.count_act_fn = nn.Softplus()
        else:
            self.count_act_fn = nn.Identity()

    def _pairwise_dist(self, a, b, eps=1e-6):
        """
        Compute pairwise distances for energy score.
        
        Args:
            a: [n, d], b: [m, d] → [n, m] of √(||a_i - b_j||² + eps)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)      # [n, m, d]
        sq   = (diff * diff).sum(-1)                # [n, m]
        return torch.sqrt(torch.clamp(sq, min=eps))

    def _compute_energy_score(self, predictions, target, λ):
        """
        Compute energy score for a single modality.
        
        Args:
            predictions: [n, m, d] - m predictions per sample
            target: [n, d] - target values
            λ: energy score regularization parameter
            
        Returns:
            energy_score: scalar loss value
        """
        n, m = predictions.shape[:2]  # Handle arbitrary dimensions after m
        
        # Flatten spatial dimensions for distance computation
        predictions_flat = predictions.view(n, m, -1)  # [n, m, d_flat]
        target_flat = target.view(n, -1)  # [n, d_flat]
        
        # Confinement term: distance to target
        target_expanded = target_flat.unsqueeze(1).expand(-1, m, -1)  # [n, m, d_flat]
        term_conf = (predictions_flat - target_expanded).norm(dim=2).mean(dim=1)  # [n]
        
        # Interaction term (efficient batched computation)
        # Using ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ identity
        sq = predictions_flat.pow(2).sum(dim=2)  # [n, m] - squared norms
        inn = torch.bmm(predictions_flat, predictions_flat.transpose(1,2))  # [n, m, m] - inner products
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2*inn  # [n, m, m] - squared distances
        sqd = torch.clamp(sqd, min=1e-6)  # avoid sqrt(0)
        d = sqd.sqrt()  # [n, m, m] - distances
        
        # Mean of off-diagonal pairwise distances
        m_mask = torch.ones(m, m, device=predictions.device) - torch.eye(m, device=predictions.device)
        mean_pd = (d * m_mask).sum(dim=(1,2)) / (m * (m - 1))  # [n]
        term_int = (λ / 2.0) * mean_pd  # [n]
        
        return (term_conf - term_int).mean()

    def forward(self, inputs, eps=None):
        """
        Forward pass through architecture.
        
        Args:
            inputs: Dict of input tensors
            eps: Optional noise tensor (if None, generate fresh noise)
            
        Returns:
            Dict with 'img' and 'counts' predictions
        """
        # Add noise (energy score specific requirement)
        inputs_with_noise = inputs.copy()
        
        # Get batch size from any tensor in inputs
        base_input = None
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                base_input = value
                break
        
        if base_input is None:
            raise ValueError("No tensor found in inputs")
            
        batch_size = base_input.shape[0]
        
        if eps is None:
            inputs_with_noise['noise'] = torch.randn(
                batch_size, self.noise_dim, device=base_input.device
            )
        else:
            inputs_with_noise['noise'] = eps
        
        # Get predictions
        raw_predictions = self.architecture(**inputs_with_noise)
        
        # Apply activation functions per modality
        predictions = {}
        if 'img' in raw_predictions:
            predictions['img'] = raw_predictions['img']  # No activation for images
        if 'counts' in raw_predictions:
            predictions['counts'] = self.count_act_fn(raw_predictions['counts'])
            
        return predictions

    def loss(self, target, inputs):
        """
        Compute multimodal energy score loss.
        
        Args:
            target: Dict with 'img' and 'counts' target values
                   - target['img']: [n, C, H, W] target images
                   - target['counts']: [n, count_dim] target count vectors
            inputs: Dict of input tensors for model
            
        Returns:
            Combined energy score loss
        """
        # Get base input to determine batch size
        def find_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, dict):
                for v in value.values():
                    result = find_tensor(v)
                    if result is not None:
                        return result
            return None
        
        base_input = None
        for value in inputs.values():
            base_input = find_tensor(value)
            if base_input is not None:
                break
                
        n = base_input.shape[0]
        λ = self.lambda_energy

        # Replicate all inputs m times
        def replicate_tensor_or_dict(value):
            if isinstance(value, torch.Tensor):
                # Expand to [n, m, ...] then reshape to [n*m, ...]
                expanded = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1))
                return expanded.reshape(n * self.m, *value.shape[1:])
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                return {k: replicate_tensor_or_dict(v) for k, v in value.items()}
            else:
                return value
        
        replicated_inputs = {}
        for key, value in inputs.items():
            replicated_inputs[key] = replicate_tensor_or_dict(value)

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=base_input.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, ...] for each modality
        raw_predictions = self.architecture(**replicated_inputs)
        
        total_loss = 0.0
        
        # Process each modality separately
        if 'img' in target and 'img' in raw_predictions:
            # Image predictions: [n*m, C, H, W] → [n, m, C, H, W]
            img_predictions = raw_predictions['img'].reshape(n, self.m, *raw_predictions['img'].shape[1:])
            img_loss = self._compute_energy_score(img_predictions, target['img'], λ)
            total_loss += self.img_weight * img_loss
            
        if 'counts' in target and 'counts' in raw_predictions:
            # Count predictions: [n*m, count_dim] → [n, m, count_dim] 
            count_predictions = self.count_act_fn(raw_predictions['counts']).reshape(n, self.m, -1)
            count_loss = self._compute_energy_score(count_predictions, target['counts'], λ)
            total_loss += self.count_weight * count_loss

        return total_loss

    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        
        Returns:
            Dict with 'img' and 'counts' predictions
        """
        predictions = self.forward(kwargs)
        
        # Post-process predictions
        result = {}
        if 'img' in predictions:
            result['img'] = predictions['img']
        if 'counts' in predictions:
            result['counts'] = predictions['counts'].round().long()
            
        return result
