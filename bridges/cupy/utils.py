import torch
import cupy as cp

def dlpack_backend(*args, backend: str = "torch"):
    if backend == "torch":
        return (torch.from_dlpack(a) for a in args)
    elif backend == "cupy":
        return (cp.from_dlpack(a) for a in args)
    else:
        raise ValueError(f"Invalid backend: {backend}")
