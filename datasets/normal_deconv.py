import torch
from torch.utils.data import Dataset
from typing import Tuple

class DeconvolutionNormalDataset(Dataset):
    """
    Each item i:
      X_i: [G, D] unit-level features (here D=4)
      Y_i: [C] group totals (here C=2), obtained by summing unit counts
    Generation:
      For each unit j and channel k:
        mean_ijk ~ Uniform, var_ijk ~ Uniform
        group bias b_{i,k} ~ Normal(0, sigma_b)
        y_{ijk} ~ Normal(mean_ijk + b_{i,k}, var_ijk)  (then clamped/rounded to >= 0)
      Y_{i,k} = sum_j y_{ijk}
      X_{ij} = [mean_{ij1}^0.8, var_{ij1}^0.7, mean_{ij2}^0.3, var_{ij2}^0.1]
    """
    def __init__(
        self,
        size: int,
        data_dim: int,
        min_value: int,
        value_range: int,
        context_dim: int,
        group_size: int = 10,
        mean_range=(5.0, 40.0),
        var_range=(2.0, 30.0),
        sigma_bias: float = 3.0,
        seed: int = 0
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n_groups = size // group_size
        G = group_size
        self.data_dim = data_dim
        self.size = size
        self.min_value = min_value
        self.value_range = value_range
        self.group_size = group_size

        means = torch.empty(n_groups, G, data_dim).uniform_(mean_range[0], mean_range[1], generator=g)
        vars_  = torch.empty(n_groups, G, data_dim).uniform_(var_range[0], var_range[1], generator=g)
        bias   = torch.normal(mean=0.0, std=sigma_bias, size=(n_groups, data_dim), generator=g)  # group-shared bias

        # unit counts then aggregate
        x_0 = torch.normal(mean=means + bias[:, None, :], std=vars_.sqrt(), generator=g)
        x_0 = torch.abs(x_0).round().clamp(min=min_value, max=min_value + value_range)                 # [B, G, data_dim]
        X_0 = x_0.sum(dim=1)                                    # [B, data_dim]


        x_1 = torch.normal(mean=torch.zeros_like(means) + (mean_range[0] + mean_range[1]) / 2, std=torch.sqrt(torch.zeros_like(vars_) + (var_range[1] - var_range[0]) / 2), generator=g)
        x_1 = torch.abs(x_1).round().clamp(min=min_value, max=min_value + value_range)                 # [B, G, data_dim]

        z = torch.stack(
            [
                means[..., i].pow(torch.rand(size=(1,), generator=g).item()) for i in range(data_dim)
            ] + 
            [
                vars_[..., i].pow(torch.rand(size=(1,), generator=g).item()) for i in range(data_dim)
            ],
            dim=-1,
        )                                                   # [B, G, 4]

        self.z = z.float()
        self.x_1 = x_1.float()
        self.x_0 = x_0.float()
        self.X_0 = X_0.float()

    def __len__(self): 
        return self.x_1.shape[0]

    def __getitem__(self, idx): 
        return {
            'x_1': self.x_1[idx],
            'x_0': self.x_0[idx],
            'z': self.z[idx],
            'X_0': self.X_0[idx]
        }
