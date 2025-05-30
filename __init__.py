"""
Count-based Flow Matching

A unified framework for discrete flow matching on count data using 
Poisson and Negative Binomial bridges.
"""

from .models import NBPosterior, BetaBinomialPosterior, MLERegressor, ZeroInflatedPoissonPosterior
# from .bridges import sample_batch_poisson, sample_batch_nb
from .datasets import (
    PoissonDataset, BetaBinomialDataset, PoissonBridgeCollate, NBBridgeCollate, 
    create_dataloader, InfiniteDataLoader
)
from .samplers import reverse_sampler
from .training import train_model, create_training_dataloader
from .scheduling import make_time_schedule, make_r_schedule, make_time_spacing_schedule

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "NBPosterior", 
    "BetaBinomialPosterior", 
    "MLERegressor", 
    "ZeroInflatedPoissonPosterior",
    # "sample_batch_poisson", 
    # "sample_batch_nb",
    "PoissonDataset",
    "BetaBinomialDataset",
    "PoissonBridgeCollate",
    "NBBridgeCollate",
    "create_dataloader",
    "InfiniteDataLoader",
    "reverse_sampler",
    "train_model",
    "create_training_dataloader",
    "make_time_schedule", 
    "make_r_schedule", 
    "make_time_spacing_schedule"
] 