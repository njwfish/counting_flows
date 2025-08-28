import torch
import torch.nn as nn

class EnergyScoreLoss(nn.Module):
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
    ):
        super().__init__()
        self.architecture = architecture
        self.noise_dim = noise_dim
        self.m = m_samples
        self.lambda_energy = lambda_energy

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
        
        return self.architecture(**inputs_with_noise)

    def loss(self, target, inputs):
        """
        Empirical energy-score (Distrib. Diffusion Models eq.14):
          L = mean_i [
            (1/m) ∑_j ||target_i - pred_ij||
            - (λ/(2(m-1))) ∑_{j≠j'} ||pred_ij - pred_ij'||
          ]
        """
        n, λ = target.shape[0], self.lambda_energy

        # Replicate all inputs m times by iterating over the dict
        replicated_inputs = {}
        for key, value in inputs.items():
            if key == 'z':
                replicated_inputs[key] = value
                continue
            replicated_inputs[key] = value.unsqueeze(1).expand(-1, self.m, *[-1] * (value.dim() - 1)).reshape(n * self.m, *value.shape[1:])

        # Add noise (energy score specific requirement)  
        noise = torch.randn(n * self.m, self.noise_dim, device=target.device)
        replicated_inputs['noise'] = noise

        # Get all predictions: [n*m, x_dim] → view [n, m, x_dim]
        predictions = self.architecture(**replicated_inputs)
        predictions = predictions.view(n, self.m, -1)  # [n, m, d]

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

    @torch.no_grad()
    def sample(self, **kwargs):
        """
        Sample prediction using arbitrary kwargs inputs.
        """
        if 'sum_constraint' in kwargs:
            S = kwargs['sum_constraint']
            del kwargs['sum_constraint']
            return self.conditional_sample(kwargs, target_sum=S)
        elif 'mean_constraint' in kwargs:
            M = kwargs['mean_constraint']
            del kwargs['mean_constraint']
            return self.conditional_sample(kwargs, target_mean=M)
        else:
            prediction = self.forward(kwargs)
            return prediction.round().long()

    @staticmethod
    def _largest_remainder_round(x, target_sum):
        """
        x: [B, D] nonnegative float; target_sum: [B] int or float
        Returns int counts y with row sums == target_sum and each |y_i - x_i| < 1.
        """
        B, D = x.shape
        device = x.device
        # Clamp negatives to zero before rounding
        x = torch.clamp(x, min=0.0)
        # Floor
        y = torch.floor(x).to(torch.int64)           # [B, D]
        # How many units still needed per row
        current = y.sum(dim=1)                       # [B]
        # If target_sum is float, round to nearest int
        tgt = torch.round(target_sum).to(torch.int64)
        rem = (tgt - current)                        # [B]
        if (rem == 0).all():
            return y
        # Fractional parts
        frac = (x - torch.floor(x))                  # [B, D]
        # For each row, add +1 to top-k fractional entries where k=rem[b]
        for b in torch.where(rem > 0)[0].tolist():
            k = int(rem[b].item())
            if k <= 0:
                continue
            # get indices of top-k fractional parts
            idx = torch.topk(frac[b], k=min(k, D), largest=True).indices
            y[b, idx] += 1
        # For rows with rem<0 (rare due to rounding), remove from smallest frac
        for b in torch.where(rem < 0)[0].tolist():
            k = int((-rem[b]).item())
            if k <= 0:
                continue
            idx = torch.topk(frac[b], k=min(k, D), largest=False).indices
            # ensure nonnegativity
            take = torch.clamp(y[b, idx], min=0)
            y[b, idx] = torch.clamp(take - 1, min=0)
        return y

    @torch.no_grad()
    def conditional_sample(
        self,
        inputs: dict,
        target_sum: torch.Tensor = None,
        target_mean: torch.Tensor = None,
        iters: int = 3,
        tol: float = 1e-6,
        tau: float = 1e-8,
        nonneg: bool = True,
        return_float: bool = False,
    ):
        """
        Minimal-norm latent correction (Gauss–Newton) to match per-row sum or mean.

        Args:
            inputs: dict of tensors for the architecture (no 'noise' key needed).
            target_sum: [B] desired sums; or
            target_mean: [B] desired means (we multiply by D).
            iters: number of correction steps.
            tol: stop if absolute sum error < tol.
            tau: small damping in denominator.
            nonneg: softplus the outputs before summing/rounding.
            return_float: if True, return the smooth outputs; else integer-rounded.

        Returns:
            y: [B, D] tensor whose row sums == target_sum (integers by default).
        """
        device = next(self.architecture.parameters()).device
        B = list(inputs.values())[0].shape[0]
        # initial noise
        eps = torch.randn(B, self.noise_dim, device=device).requires_grad_(True)

        # we'll compute per-sample corrections; use small loop for clarity
        for _ in range(iters):
            x = self._gen_with_noise(inputs, eps, nonneg=nonneg)      # [B, D]
            Ddim = x.shape[-1]
            if target_sum is None:
                if target_mean is None:
                    raise ValueError("Provide target_sum or target_mean.")
                tgt = target_mean.to(x).flatten() * Ddim
            else:
                tgt = target_sum.to(x).flatten()
            g = (x.sum(dim=1) - tgt)                                  # [B]
            if torch.all(g.abs() <= tol):
                break
            # compute v_b = ∇_{eps_b} g_b for each b
            v_list = []
            eps_grad = torch.zeros_like(eps)
            for b in range(B):
                if eps.grad is not None:
                    eps.grad.zero_()
                gb = g[b]
                gb.backward(retain_graph=True)
                v_b = eps.grad[b].clone()                              # [noise_dim]
                v_list.append(v_b)
                eps.grad.zero_()
            v = torch.stack(v_list, dim=0)                             # [B, noise_dim]
            denom = (v.pow(2).sum(dim=1) + tau).unsqueeze(1)           # [B, 1]
            step = (g / denom.squeeze(1)).unsqueeze(1) * v             # [B, noise_dim]
            eps = (eps - step).detach().requires_grad_(True)

        x = self._gen_with_noise(inputs, eps, nonneg=nonneg)           # [B, D]
        # exact integerization with preserved sums
        if return_float:
            return x
        y = self._largest_remainder_round(x, tgt)
        return y

    @torch.no_grad()
    def conditional_sparse_sample(
        self,
        inputs: dict,
        target_sum: torch.Tensor = None,
        target_mean: torch.Tensor = None,
        max_steps: int = 3,
        cap: float = None,
        tol: float = 1e-6,
        nonneg: bool = True,
        return_float: bool = False,
    ):
        """
        Sparse ℓ1-style latent correction: greedy one-coordinate updates per sample.

        Args:
            inputs: dict of tensors for the architecture (no 'noise' key needed).
            target_sum / target_mean: same semantics as above.
            max_steps: greedy relinearizations (often 1–3 is enough).
            cap: optional max absolute change per coordinate in noise (safety).
            tol: stop if absolute sum error < tol.
            nonneg: apply softplus before summing/rounding.
            return_float: return smooth outputs if True.

        Returns:
            y: [B, D] tensor (integers by default) with exact target sums.
        """
        device = next(self.architecture.parameters()).device
        B = list(inputs.values())[0].shape[0]
        eps = torch.randn(B, self.noise_dim, device=device).requires_grad_(True)

        for _ in range(max_steps):
            x = self._gen_with_noise(inputs, eps, nonneg=nonneg)      # [B, D]
            Ddim = x.shape[-1]
            if target_sum is None:
                if target_mean is None:
                    raise ValueError("Provide target_sum or target_mean.")
                tgt = target_mean.to(x).flatten() * Ddim
            else:
                tgt = target_sum.to(x).flatten()
            g = (x.sum(dim=1) - tgt)                                  # [B]
            if torch.all(g.abs() <= tol):
                break
            # coordinate-wise update
            for b in range(B):
                # backprop scalar g_b
                if eps.grad is not None:
                    eps.grad.zero_()
                gb = g[b]
                gb.backward(retain_graph=True)
                v = eps.grad[b]                                        # [noise_dim]
                k = torch.argmax(v.abs()).item()
                vk = v[k].detach()
                if vk.abs() < 1e-12:
                    continue
                step = (-gb.detach() / (vk + 1e-12))
                if cap is not None:
                    step = torch.clamp(step, min=-abs(cap), max=abs(cap))
                with torch.no_grad():
                    eps[b, k] += step
                eps.grad.zero_()
            eps = eps.detach().requires_grad_(True)

        x = self._gen_with_noise(inputs, eps, nonneg=nonneg)
        if return_float:
            return x
        y = self._largest_remainder_round(x, tgt)
        return y