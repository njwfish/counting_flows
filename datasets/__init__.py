"""Dataset modules for counting flows"""
from .mnist import MNIST
from .gaussian_mixture import GaussianMixtureDataset, LowRankGaussianMixtureDataset

__all__ = [
    'MNIST',
    'GaussianMixtureDataset', 
    'LowRankGaussianMixtureDataset',
]

# Optional imports that may not be available
try:
    from .discrete_moons import DiscreteMoonsDataset
    __all__.append('DiscreteMoonsDataset')
except ImportError:
    pass
