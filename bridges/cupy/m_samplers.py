import cupy as cp

class PoissonM:
    def __init__(self, lam_p: float, lam_m: float):
        self.lam_p = lam_p
        self.lam_m = lam_m
        self.lam_star = 2.0 * cp.sqrt(self.lam_p * self.lam_m)
        self.markov = False
    
    def __call__(self, diff: cp.ndarray):
        return cp.random.poisson(self.lam_star, diff.shape)

class SkellamM:
    def __init__(self, lam_p: float, lam_m: float):
        self.lam_p = lam_p
        self.lam_m = lam_m
        self.lam_star = 2.0 * cp.sqrt(self.lam_p * self.lam_m)
        self.markov = True
        
    def __call__(self, diff: cp.ndarray):
        raise NotImplementedError("SkellamM is not implemented yet")