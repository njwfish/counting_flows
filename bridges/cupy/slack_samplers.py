import cupy as cp

from bridges.cupy.sampling.bessel import bessel

class PoissonM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = False):
        self.lam_p = cp.asarray(lam_p)
        self.lam_m = cp.asarray(lam_m)
        self.lam_star = 2.0 * cp.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
    
    def __call__(self, diff: cp.ndarray):
        return cp.random.poisson(self.lam_star, diff.shape)

class BesselM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = True):
        self.lam_p = cp.asarray(lam_p)
        self.lam_m = cp.asarray(lam_m)
        self.lam_star = 2.0 * cp.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
        
    def __call__(self, diff: cp.ndarray):
        lam_p = cp.broadcast_to(self.lam_p, diff.shape)
        lam_m = cp.broadcast_to(self.lam_m, diff.shape)
        
        return bessel(lam_p, lam_m, diff)