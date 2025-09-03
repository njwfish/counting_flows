import torch
import os   
import logging
import hashlib
import json
from omegaconf import DictConfig, OmegaConf

# Enable JIT compilation for better performance
# TODO: We are using DDP 
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True

USE_COMPILE = False
try:

    if os.environ.get("USE_COMPILE") == "1":
        USE_COMPILE = True
 
except Exception as e:
    USE_COMPILE = False

logger = logging.getLogger(__name__)

logger.info(f"Using compile: {USE_COMPILE}")

# Try to use torch.compile if available (PyTorch 2.0+)
def maybe_compile(func):

    """Decorator to optionally compile functions for better performance."""
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            return torch.compile(func, mode='max-autotune')
        except:
            pass
    return func


def get_model_hash(cfg: DictConfig) -> str:
    """
    Get model hash excluding training and sampling parameters.
    Only includes model-relevant configuration for consistent model identification.
    """
    # Create config hash (excluding training and sampling params)
    config_copy = OmegaConf.to_container(cfg, resolve=True)
    
    # Remove entire training section (not model-relevant)
    config_copy.pop('training', None)
    
    # Remove other non-model-relevant sections and parameters
    excluded_params = [
        # Hydra/experiment management
        'hydra', 'defaults', 'logging',
        # Runtime parameters  
        'device', 'create_plots',
        # Sampling parameters
        'n_steps', 'n_samples'
    ]
    
    for param in excluded_params:
        config_copy.pop(param, None)
    
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]