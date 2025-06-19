"""
Count-based Flow Matching

A unified framework for discrete flow matching on count data using 
Poisson and Negative Binomial bridges.
"""

from .models import EnergyScorePosterior
from .datasets import PoissonDataset, BetaBinomialDataset, create_dataloader, InfiniteDataLoader
from .bridges.skellam import SkellamBridge
from .bridges.reflected import ReflectedPoissonBDBridge
from .bridges.constrained import SkellamMeanConstrainedBridge
from .training import train_model, create_training_dataloader
from .scheduling import make_time_spacing_schedule

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "EnergyScorePosterior", 
    "PoissonDataset",
    "BetaBinomialDataset",
    "SkellamBridge",
    "ReflectedPoissonBDBridge", 
    "SkellamMeanConstrainedBridge",
    "create_dataloader",
    "InfiniteDataLoader",
    "train_model",
    "create_training_dataloader",
    "make_time_spacing_schedule"
] 