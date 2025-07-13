"""
Count-based Flow Matching

A unified framework for discrete flow matching on count data using 
Poisson and Negative Binomial bridges.
"""

from counting_flows.models import EnergyScorePosterior
from counting_flows.datasets import PoissonDataset, BetaBinomialDataset, create_dataloader, InfiniteDataLoader
from counting_flows.bridges.numpy.skellam import SkellamBridge
from counting_flows.bridges.numpy.constrained import SkellamMeanConstrainedBridge
from counting_flows.training import train_model, create_training_dataloader
# from counting_flows.bridges.torch.scheduling import make_time_spacing_schedule

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "EnergyScorePosterior", 
    "PoissonDataset",
    "BetaBinomialDataset",
    "SkellamBridge",
    "SkellamMeanConstrainedBridge",
    "create_dataloader",
    "InfiniteDataLoader",
    "train_model",
    "create_training_dataloader",
    "make_time_spacing_schedule"
] 