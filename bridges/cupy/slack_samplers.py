import cupy as cp

from .sampling.bessel import bessel

class ConstantM:
    def __init__(self, m: int, markov: bool = True):
        self.m = m
        self.markov = markov

    def __call__(self, diff: cp.ndarray, w: cp.ndarray):
        return cp.round(cp.full(diff.shape, self.m) * w)

class BesselM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = True):
        self.lam_p = cp.asarray(lam_p)
        self.lam_m = cp.asarray(lam_m)
        self.markov = markov
        
    def __call__(self, diff: cp.ndarray, w: cp.ndarray):
        lam_p = cp.broadcast_to(self.lam_p, diff.shape) * w
        lam_m = cp.broadcast_to(self.lam_m, diff.shape) * w
        
        return bessel(lam_p, lam_m, diff)
