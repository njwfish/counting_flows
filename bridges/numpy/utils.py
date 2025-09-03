import torch
import numpy as np
from typing import Optional, Iterable
import cupy as cp

def dlpack_backend(*args, backend: str = "torch", dtype: Optional = None, device: Optional = None):
    if not isinstance(args, Iterable):
        args = (args,)
    if backend == "torch":
        dtype = torch.float32 if dtype == "float32" else dtype
        return tuple(torch.from_numpy(a) if dtype is None else torch.from_numpy(a).to(dtype).to(device) for a in args)
    elif backend == "numpy":
        dtype = np.int32 if dtype == "int32" else dtype
        # if is torch, convert to numpy
        if isinstance(args[0], torch.Tensor):
            args = tuple(a.cpu().numpy() for a in args)
        elif isinstance(args[0], np.ndarray):
            args = tuple(a.astype(dtype) if dtype is not None else a for a in args)
        elif isinstance(args[0], cp.ndarray):
            args = tuple(a.astype(dtype) if dtype is not None else a for a in args)
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return args if len(args) > 1 else args[0]
    else:
        raise ValueError(f"Invalid backend: {backend}") 