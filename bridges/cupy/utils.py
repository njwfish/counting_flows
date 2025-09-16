import torch
import cupy as cp
from typing import Optional, Iterable

from_numpy = {
    'cupy': cp.array,
    'torch': torch.from_numpy,
    'numpy': np.array,
}

def dlpack_backend(*args, backend: str = "torch", dtype: Optional = None):
    if not isinstance(args, Iterable):
        args = (args,)

    if isinstance(args[0], cp.ndarray):
        return tuple(from_numpy[backend](a) for a in args)

    if backend == "torch":
        dtype = torch.float32 if dtype == "float32" else dtype
        return tuple(torch.from_dlpack(a) if dtype is None else torch.from_dlpack(a).to(dtype) for a in args)
    elif backend == "cupy":
        dtype = cp.int32 if dtype == "int32" else dtype
        return tuple(cp.from_dlpack(a) if dtype is None else cp.from_dlpack(a).astype(dtype) for a in args)
    else:
        raise ValueError(f"Invalid backend: {backend}")
