import torch


def concat_inputs(**kwargs):
    """
    Simple utility to concatenate all input tensors for architectures.
    
    Args:
        **kwargs: Named tensors to concatenate
        
    Returns:
        torch.Tensor: Concatenated tensor along last dimension
    """
    # Get all tensor values, maintaining consistent order by sorting keys
    tensors = [kwargs[key] for key in sorted(kwargs.keys())]
    return torch.cat(tensors, dim=-1) 