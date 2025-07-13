import torch
import os   

# Enable JIT compilation for better performance
# TODO: We are using DDP 
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True

try:

    if os.environ.get("USE_COMPILE") == "1":
        USE_COMPILE = True
 
except:
    USE_COMPILE = False

# Try to use torch.compile if available (PyTorch 2.0+)
def maybe_compile(func):

    """Decorator to optionally compile functions for better performance."""
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            return torch.compile(func, mode='max-autotune')
        except:
            pass
    return func