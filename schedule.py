import torch
import math

def create_cosine_warmup_constant_schedule(warmup_steps, start_factor=0.01):
    """
    Returns a lambda function for use with LambdaLR.
    Much simpler than inheriting from _LRScheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Cosine warmup
            progress = step / warmup_steps
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            return start_factor + (1.0 - start_factor) * cosine_factor
        else:
            # Constant
            return 1.0
    
    return lr_lambda

# Factory function for Hydra
def cosine_warmup_constant_lr(optimizer, warmup_steps, start_factor=0.01):
    """Factory function that creates a LambdaLR scheduler"""
    lr_lambda = create_cosine_warmup_constant_schedule(warmup_steps, start_factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)