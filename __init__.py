"""
Count-based Flow Matching

A unified framework for discrete flow matching on count data using 
Poisson and Negative Binomial bridges.
"""

from .models import EnergyScorePosterior
from .datasets import PoissonDataset, BetaBinomialDataset, create_dataloader, InfiniteDataLoader
from .bridges.skellam import SkellamBridge
from .bridges.reflected import ReflectedSkellamBridge
from .bridges.constrained import SkellamMeanConstrainedBridge
from .training import train_model, create_training_dataloader
from .bridges.scheduling import make_time_spacing_schedule

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "EnergyScorePosterior", 
    "PoissonDataset",
    "BetaBinomialDataset",
    "SkellamBridge",
    "ReflectedSkellamBridge", 
    "SkellamMeanConstrainedBridge",
    "create_dataloader",
    "InfiniteDataLoader",
    "train_model",
    "create_training_dataloader",
    "make_time_spacing_schedule"
] 