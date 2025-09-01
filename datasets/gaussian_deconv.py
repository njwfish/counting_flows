import torch
from torch.utils.data import Dataset
from torch.distributions import Dirichlet
from typing import Tuple, Dict, Any
import numpy as np
from .gaussian_mixture import GaussianMixtureDataset, LowRankGaussianMixtureDataset
from .mnist import DiffMNIST

class DeconvolutionGaussianMixtureDataset(Dataset):
    """
    Deconvolution dataset that wraps LowRankGaussianMixtureDataset.
    
    This dataset:
    1. Uses LowRankGaussianMixtureDataset as the underlying data source
    2. Provides one-hot context vectors indicating mixture component
    3. Uses Dirichlet sampling to determine component mixing in groups
    4. Samples actual group members from the appropriate mixture components
    
    Each item returns:
      x_0: Group of source samples from Gaussian mixture [group_size, data_dim]
      x_1: Group of target samples from Gaussian mixture [group_size, data_dim] 
      z: One-hot matrix indicating component for each group member [group_size, k]
      X_0: Sum of group members [data_dim]
      
    The key innovation is that x_0/X_0 contains actual samples from the dataset, with the number
    of samples from each mixture component determined by Dirichlet-sampled weights.
    """
    def __init__(
        self,
        base_size: int,
        data_dim: int,
        min_value: int,
        value_range: int,
        context_dim: int,  # This should equal k (number of mixture components)
        group_size: int = 10,
        latent_dim: int = None,  # For low-rank variant
        k: int = 5,  # Number of mixture components
        dirichlet_concentration: float = 1.0,  # Concentration parameter for Dirichlet
        seed: int = 42,
        **kwargs  # Additional args passed to LowRankGaussianMixtureDataset
    ):
        super().__init__()
        self.size = base_size // group_size
        self.base_size = base_size
        self.data_dim = data_dim
        self.min_value = min_value
        self.value_range = value_range
        self.group_size = group_size
        self.k = k
        self.dirichlet_concentration = dirichlet_concentration
        
        # Use low-rank variant if latent_dim is specified, otherwise use regular variant
        if latent_dim is not None:
            self.mixture_dataset = LowRankGaussianMixtureDataset(
                size=base_size,
                data_dim=data_dim,
                latent_dim=latent_dim,
                value_range=value_range,
                min_value=min_value,
                k=k,
                seed=seed,
                **kwargs
            )
        else:
            self.mixture_dataset = GaussianMixtureDataset(
                size=base_size,
                data_dim=data_dim,
                value_range=value_range,
                min_value=min_value,
                k=k,
                seed=seed,
                **kwargs
            )
        
        # Create component index lists and pre-compute all groups
        self._create_component_indices()
        self._pre_compute_all_groups(seed)
        
    def _create_component_indices(self):
        """Create lists of indices for each mixture component"""
        self.component_indices = [[] for _ in range(self.k)]
        
        # Group indices by their component assignment
        for idx in range(self.size):
            component = self.mixture_dataset.x0_components[idx].item()
            self.component_indices[component].append(idx)
            
    def _pre_compute_all_groups(self, seed: int):
        """Pre-compute all groups using Dirichlet weights"""
        # Use torch's random state for consistent seeding
        with torch.random.fork_rng():
            torch.manual_seed(seed)  # Different seed to avoid conflicts
            
            # Sample Dirichlet weights for component allocation
            # Shape: [size, k] - each row sums to 1, represents component proportions
            dirichlet_dist = Dirichlet(self.dirichlet_concentration * self.mixture_dataset.weights_source * self.k)
            component_weights = dirichlet_dist.sample((self.size,))
            
            # Pre-allocate storage for all groups and their component info
            self.groups = torch.zeros(self.size, self.group_size, self.data_dim, dtype=torch.float32)
            self.group_component_info = torch.zeros(self.size, self.group_size, dtype=torch.int64)
            self.group_one_hot = torch.zeros(self.size, self.group_size, self.k, dtype=torch.float32)
            
            # Create each group
            for group_idx in range(self.size):
                # Convert weights to actual counts for this group
                counts = torch.multinomial(
                    component_weights[group_idx], 
                    self.group_size, 
                    replacement=True
                )
                # Count how many from each component
                component_counts = torch.bincount(counts, minlength=self.k)
                
                # Sample actual group members from each component
                group_members = []
                group_components = []
                for component_id in range(self.k):
                    count = component_counts[component_id].item()
                    if count > 0:
                        # Sample 'count' members from this component's indices (with replacement)
                        available_indices = self.component_indices[component_id]
                        if len(available_indices) > 0:
                            sampled_indices = torch.randint(
                                0, len(available_indices), (count,)
                            )
                            
                            # Get their x_0 values and store component info
                            for sampled_idx in sampled_indices:
                                member_idx = available_indices[sampled_idx.item()]
                                member_sample = self.mixture_dataset[member_idx]
                                group_members.append(member_sample['x_0'])
                                group_components.append(component_id)
                
                # Store the group and component info
                # Note: with sufficient samples, we should always get exactly group_size members
                self.groups[group_idx] = torch.stack(group_members)
                self.group_component_info[group_idx] = torch.tensor(group_components)
                
                # Pre-compute one-hot vectors for each group member
                for member_idx, component_id in enumerate(group_components):
                    self.group_one_hot[group_idx, member_idx, component_id] = 1.0

    def __len__(self): 
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        # Get random target sample (same as before)
        target_idx = np.random.randint(0, self.size, self.group_size)
        x_1 = self.mixture_dataset.x1_data[target_idx]  # [group_size, data_dim]
        
        # Get pre-computed group as both x_0 and X_0
        x_0 = self.groups[idx]  # [group_size, data_dim]
        X_0 = x_0.sum(dim=0)  # [data_dim] - sum of group members
        
        # Get pre-computed one-hot vectors for each group member
        z = self.group_one_hot[idx]  # [group_size, k] - one-hot for each member
        
        return {
            'x_0': x_0.float(),  # [group_size, data_dim]
            'x_1': x_1.float(),  # [group_size, data_dim] 
            'z': z.float(),      # [group_size, k] - one-hot for each member
            'X_0': X_0.float()   # [data_dim] - sum of group members
        }


class DeconvolutionMNISTGaussianMixtureDataset(DeconvolutionGaussianMixtureDataset):
    """
    Deconvolution dataset that combines MNIST images with Gaussian mixture data.
    
    This dataset extends DeconvolutionGaussianMixtureDataset by:
    1. Using MNIST images selected based on mixture component assignments
    2. Providing structured x_0/x_1 with both 'img' (MNIST) and 'counts' (Gaussian) components
    3. Maintaining the same group-based structure for deconvolution
    
    Each item returns:
      x_0: Dict with 'img' and 'counts' components [group_size, ...]
      x_1: Dict with 'img' (noise) and 'counts' (Gaussian) [group_size, ...]
      z: One-hot matrix indicating component for each group member [group_size, k]
      X_0: Sum of Gaussian counts [data_dim]
    """
    
    def __init__(
        self,
        mnist_data_dir: str = "datasets/data/mnist",
        **kwargs
    ):
        """
        Args:
            mnist_data_dir: Directory for MNIST data
            **kwargs: Arguments passed to parent DeconvolutionGaussianMixtureDataset
        """
        super().__init__(**kwargs)
        
        # Load MNIST dataset
        self.mnist_dataset = DiffMNIST(mnist_data_dir)
        
        # Group MNIST data by digit labels (0-9) to align with mixture components
        self.mnist_by_digit = {}
        for digit in range(10):
            digit_indices = (self.mnist_dataset.labels == digit).nonzero(as_tuple=True)[0]
            self.mnist_by_digit[digit] = digit_indices
        
        # Pre-compute MNIST selections for each group based on component assignments
        self._pre_compute_mnist_selections()
    
    def _pre_compute_mnist_selections(self):
        """Pre-compute MNIST image selections based on mixture component assignments"""
        self.mnist_selections = torch.zeros(self.size, self.group_size, dtype=torch.long)
        
        for group_idx in range(self.size):
            # Get component assignments for this group
            component_assignments = self.group_component_info[group_idx]
            
            for member_idx, component_id in enumerate(component_assignments):
                # Use component_id as MNIST digit (mod 10 for safety)
                digit = component_id.item() % 10
                
                # Randomly select an MNIST image of this digit
                digit_indices = self.mnist_by_digit[digit]
                if len(digit_indices) > 0:
                    selected_idx = digit_indices[torch.randint(0, len(digit_indices), (1,))]
                    self.mnist_selections[group_idx, member_idx] = selected_idx
                else:
                    # Fallback: random MNIST image
                    self.mnist_selections[group_idx, member_idx] = torch.randint(
                        0, len(self.mnist_dataset), (1,)
                    )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the base Gaussian mixture data
        base_data = super().__getitem__(idx)
        
        # Get MNIST images for this group
        mnist_indices = self.mnist_selections[idx]  # [group_size]
        mnist_images = torch.stack([
            self.mnist_dataset.data[idx] for idx in mnist_indices
        ])  # [group_size, 1, 28, 28]
        
        # Generate noise for x_1 MNIST component (same as DiffMNIST)
        mnist_noise = torch.randn_like(mnist_images)  # [group_size, 1, 28, 28]
        
        # Structure the data with nested dictionaries
        return {
            'x_0': {
                'img': mnist_images.float(),      # [group_size, 1, 28, 28]
                'counts': base_data['x_0']        # [group_size, data_dim] 
            },
            'x_1': {
                'img': mnist_noise.float(),       # [group_size, 1, 28, 28]
                'counts': base_data['x_1']        # [group_size, data_dim]
            },
            'X_0': base_data['X_0']               # [data_dim] - sum of Gaussian counts only
        }
