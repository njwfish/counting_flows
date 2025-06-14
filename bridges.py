import torch
from torch.distributions import NegativeBinomial
import numpy as np
from .scheduling import make_time_spacing_schedule, make_lambda_schedule
from torch.distributions import Multinomial

def manual_hypergeometric(total_count, num_successes, num_draws):
    """
    Vectorized implementation of hypergeometric sampling using numpy broadcasting.
    
    Args:
        total_count: Total population size (array-like)
        num_successes: Number of success items in population (array-like)
        num_draws: Number of items drawn (array-like)
    
    Returns:
        Number of successes in the drawn sample (same shape as inputs)
    """
    # Convert to numpy arrays for vectorized operations
    total_count = np.asarray(total_count)
    num_successes = np.asarray(num_successes)
    num_draws = np.asarray(num_draws)
    
    # Bounds checking to prevent invalid hypergeometric parameters
    num_successes = np.clip(num_successes, 0, total_count)
    num_draws = np.clip(num_draws, 0, total_count)
    
    # Handle edge cases
    mask_zero_draws = (num_draws == 0)
    mask_zero_successes = (num_successes == 0)
    mask_draws_ge_total = (num_draws >= total_count)
    mask_successes_ge_total = (num_successes >= total_count)
    
    # Use numpy's vectorized hypergeometric
    result = np.zeros_like(num_draws)
    
    # Only call hypergeometric for valid cases
    valid_mask = ~(mask_zero_draws | mask_zero_successes | mask_draws_ge_total | mask_successes_ge_total)
    
    if np.any(valid_mask):
        result[valid_mask] = np.random.hypergeometric(
            num_successes[valid_mask], 
            total_count[valid_mask] - num_successes[valid_mask], 
            num_draws[valid_mask]
        )
    
    # Handle edge cases
    result[mask_zero_draws] = 0
    result[mask_zero_successes] = 0
    result[mask_draws_ge_total] = num_successes[mask_draws_ge_total]
    result[mask_successes_ge_total] = num_draws[mask_successes_ge_total]
    
    return result

# -----------------------------------------------------------------------------
# 1) Sum‐constrained Multinomial proposal
# -----------------------------------------------------------------------------
def constrained_multinomial_proposal(x: torch.LongTensor,
                                     x_exp: torch.Tensor,
                                     S: torch.LongTensor):
    B, d = x.shape
    R     = S - x.sum(dim=0)          # (d,)
    sgn  = R.sign().long()            # in {-1,0,1}
    Rabs = R.abs().long()             # integer total_count per sample

    diff     = sgn.unsqueeze(0) * (x_exp - x).float()  # (B,d)
    weights  = diff.clamp(min=0.0)
    wsum     = weights.sum(dim=1, keepdim=True)
    weights += (wsum == 0).float()          # (B,d)
    wsum     = weights.sum(dim=1, keepdim=True)
    probs    = weights / wsum          # (B,d)

    # draw delta[b] ~ Multinomial(Rabs[b], probs[b])
    delta = []
    for b in range(d):
        if Rabs[b].item() > 0:
            m_b = Multinomial(total_count=Rabs[b].item(),
                              probs=probs[:, b]).sample()
        else:
            m_b = torch.zeros(B, device=x.device)
        delta.append(m_b)
    delta = torch.stack(delta, dim=1).round().long()  # (B,d)

    x_prop = x + sgn.unsqueeze(0) * delta
    return x_prop, sgn


# -----------------------------------------------------------------------------
# 2) Hypergeometric log‐pmf
# -----------------------------------------------------------------------------
def hypergeom_logpmf(k: torch.LongTensor,
                    N: torch.LongTensor,
                    K: torch.LongTensor,
                    n: torch.LongTensor) -> torch.Tensor:
    """
    Log‐pmf of Hypergeom(total=N, successes=K, draws=n) at k:
      pmf(k) = C(K,k)*C(N-K,n-k) / C(N,n)
    All inputs are LongTensors of same shape.
    """
    return (
        torch.lgamma(K+1) - torch.lgamma(k+1) - torch.lgamma(K-k+1)
      + torch.lgamma(N-K+1)
        - torch.lgamma((n-k)+1)
        - torch.lgamma((N-K)-(n-k)+1)
      - (torch.lgamma(N+1) - torch.lgamma(n+1) - torch.lgamma(N-n+1))
    )

# -----------------------------------------------------------------------------
# 3) MH sweep for mean‐constrained batch reverse‐kernel
# -----------------------------------------------------------------------------
@torch.no_grad()
def mh_mean_constrained_update(
    x:       torch.LongTensor,
    x_exp:   torch.Tensor,
    S:       torch.LongTensor,
    a:       torch.LongTensor,
    N_t:     torch.LongTensor,
    B_t:     torch.LongTensor,
    N_s:     torch.LongTensor,
    sweeps:  int = 10
) -> torch.LongTensor:
    """
    Perform `sweeps` Metropolis‐Hastings “swap” updates to sample from
    the batch‐mean‐constrained reverse kernel:
      \tilde P(\mathbf y | \mathbf x)
      ∝ ∏_i Hypergeom(N_t[i], B_t[i], N_s[i])  subject to sum(y)=S.

    Args:
      x      (B,d)   current counts at time s (sum must = S)
      x_exp  (B,d)   E[x_i] from forward model (for proposal weights)
      S      (B,)    target sum per batch sample
      a      (B,d)   rounded x0_hat from model at time s
      N_t    (B,d)   latent total jumps at time t
      B_t    (B,d)   latent births at time t
      N_s    (B,d)   latent total jumps at time s
      sweeps int     number of MH sweeps
    Returns:
      x      (B,d)   new counts at time s (sum = S)
    """
    B, d = x.shape
    device = x.device
    idx = torch.arange(B, device=device)

    # 0) ensure we start on the correct manifold
    x, _ = constrained_multinomial_proposal(x, x_exp, S)

    for _ in range(sweeps):
        # 1) pick random pair (i->+1, j->-1)
        i = torch.randint(d, (B,), device=device)
        j = torch.randint(d, (B,), device=device)   

        # 2) propose by swapping one unit
        x_prop = x.clone()
        x_prop[idx, i] += 1
        x_prop[idx, j] -= 1
        # x_prop = x_prop.clamp(min=0)

        # 3) implied B_s for proposed & current
        #    B_s = (N_s + (x - a)) // 2
        Bs_prop = ((N_s + (x_prop - a)) >> 1)
        Bs_curr = ((N_s + (x      - a)) >> 1)

        # 4) log‐pmf under product of two Hypergeom (coords i,j)
        def pick(tensor, coords):
            return tensor[idx, coords]
        Bsp_i = pick(Bs_prop, i); Bsp_j = pick(Bs_prop, j)
        Bsc_i = pick(Bs_curr, i); Bsc_j = pick(Bs_curr, j)
        Nti   = pick(N_t,    i); Ntj   = pick(N_t,    j)
        Bti   = pick(B_t,    i); Btj   = pick(B_t,    j)
        Nsi   = pick(N_s,    i); Nsj   = pick(N_s,    j)

        logp_prop = (
            hypergeom_logpmf(Bsp_i, Nti, Bti, Nsi)
          + hypergeom_logpmf(Bsp_j, Ntj, Btj, Nsj)
        )
        logp_curr = (
            hypergeom_logpmf(Bsc_i, Nti, Bti, Nsi)
          + hypergeom_logpmf(Bsc_j, Ntj, Btj, Nsj)
        )

        # 5) MH accept
        log_alpha = logp_prop - logp_curr
        alpha     = log_alpha.exp().clamp(max=1.0)
        u         = torch.rand_like(alpha)
        accept    = (u < alpha)

        # 6) apply accepted proposals
        x[accept] = x_prop[accept]

    return x


class PoissonBDBridgeCollate:
    """
    Collate for Poisson birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,
        lam_p1: float = 8.0,
        lam_m0: float = 8.0,
        lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.sched_kwargs = schedule_kwargs
        
        # Create time spacing
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

    def _sample_N_B1(self, diff: torch.Tensor, Lambda_p: torch.Tensor, Lambda_m: torch.Tensor, device):
        """
        diff = x1 - x0   (shape [B, d])
        Lambda_p, Lambda_m: Final cumulative integrals from lambda schedule (scalars)
        Returns N (total jumps) and B1 (birth count at t=1).
        Uses Poisson for total jumps with lambda_star from schedule.
        """
        B, d = diff.shape
        # Use the cumulative lambda integrals
        lambda_star = 2.0 * torch.sqrt(Lambda_p * Lambda_m)        # scalar
        lambda_star = lambda_star.unsqueeze(-1).expand_as(diff)    # shape (B, d)
        M = torch.poisson(lambda_star).long().to(device)
        N = diff.abs() + 2 * M
        B1 = (N + diff) // 2   # integer division
        return N, M, B1

    def _sample_time(self):
        """Sample time point from pre-computed schedule"""
        # Sample from valid time points (excluding endpoints)
        valid_idx = torch.randint(1, len(self.time_points) - 1, (1,)).item()
        t = self.time_points[valid_idx].item()
        t_idx = valid_idx
        return t, t_idx

    def __call__(self, batch):
        # batch = list of dicts with keys 'x0','x1','z'
        x0 = torch.stack([b["x0"] for b in batch])  # (1,B,d) -> squeeze to (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])
        
        # Handle case where dataset already produces batches (extra dimension from DataLoader)
        if len(x0.shape) == 3:  # (1, B, d)
            x0 = x0.squeeze(0)  # (B, d)
            x1 = x1.squeeze(0)  # (B, d)
            z = z.squeeze(0)    # (B, context_dim)
        
        device = x0.device
        B, d = x0.shape

        # schedules
        grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
            self.n_steps, self.lam_p0, self.lam_p1, self.lam_m0, self.lam_m1,
            self.schedule_type, device=device
        )
        Λ_tot1 = Λp[-1] + Λm[-1]

        # latent (N,B1)
        diff = (x1 - x0)          # (B, d) - don't squeeze
        N, M, B1 = self._sample_N_B1(diff, Λp[-1], Λm[-1], device)       # shape (B, d)

        # pick random interior k (avoid k=0,K)
        if self.homogeneous_time:
            # Sample one time index for entire batch
            k_idx_scalar = torch.randint(1, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_scalar, device=device)
        else:
            # Sample different time indices for each sample (original behavior)
            k_idx = torch.randint(1, self.n_steps + 1, (B,), device=device)
            
            
        t     = grid[k_idx]                               # (B,)
        w_t   = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1          # (B,)
        N_t   = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()       # (B, d)

        # draw B_t ~ Hypergeom(N, B1, N_t)
        B_t = torch.empty_like(N)
        
        # Vectorized hypergeometric sampling
        B_t_np = manual_hypergeometric(
            total_count=N.cpu().numpy(),
            num_successes=B1.cpu().numpy(),
            num_draws=N_t.cpu().numpy()
        )
        B_t = torch.from_numpy(B_t_np).to(device)

        x_t = x0 + (2 * B_t - N_t)          # (B, d)

        diff_t = (x_t - x0).abs()          # (B,d)
        M_t    = ((N_t - diff_t) >> 1)     # integer tensor (B,d)

        return {
            "x0"  : x0,
            "x1"  : x1,
            "x_t" : x_t,
            "t"   : t,
            "z"   : z,
            "N"   : N,
            "M" : M_t, 
            "B1"  : B1,
            "grid": grid,
            "Λp"  : Λp,
            "Λm"  : Λm,
        }


class ReflectedPoissonBDBridgeCollate:
    """
    Collate for *reflected* Poisson birth–death bridge (equal λ₊,λ₋ at each time).
    Samples x_t = |X_t| by drawing X_t via Hypergeometric(N, B1, N_t) then taking abs.
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,
        lam_p1: float = 8.0,
        lam_m0: float = 8.0,
        lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.lam_p0 = lam_p0
        self.lam_p1 = lam_p1
        self.lam_m0 = lam_m0
        self.lam_m1 = lam_m1
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        # precompute time grid for optional use
        self.time_points = make_time_spacing_schedule(n_steps, time_schedule, **schedule_kwargs)

    def _sample_N_B1(self, diff: torch.LongTensor, Lambda_p: torch.Tensor, Lambda_m: torch.Tensor, device):
        """
        diff = x1 - x0   (LongTensor [B, d])
        Lambda_p, Lambda_m: scalars (cumulative integrals at t=1)
        Returns:
          N  : LongTensor total jumps by t=1
          B1 : LongTensor births by t=1
        """
        # scalar λ*
        lambda_star = 2.0 * torch.sqrt(Lambda_p * Lambda_m)          # float scalar
        lambda_star = lambda_star.unsqueeze(-1).expand_as(diff)      # (B,d)
        M = torch.poisson(lambda_star).long().to(device)             # over‐dispersion
        N = diff.abs() + 2 * M                                       # total jumps
        B1 = ((N + diff) >> 1)                                       # births
        return N, B1

    def __call__(self, batch):
        # unpack batch of dicts
        x0 = torch.stack([b["x0"] for b in batch])
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])

        # handle extra singleton batch‐dim
        if x0.dim() == 3:
            x0 = x0.squeeze(0)
            x1 = x1.squeeze(0)
            z  = z .squeeze(0)

        device = x0.device
        B, d = x0.shape

        # 1) build forward‐time schedules
        grid, lam_p, lam_m, Λp, Λm = make_lambda_schedule(
            self.n_steps,
            self.lam_p0, self.lam_p1,
            self.lam_m0, self.lam_m1,
            self.schedule_type,
            device=device
        )
        Λ_tot1 = Λp[-1] + Λm[-1]

        # 2) sample latent (N, B1) at t=1
        diff = (x1 - x0).round().long()  # ensure integer
        N, B1 = self._sample_N_B1(diff, Λp[-1], Λm[-1], device)

        # 3) choose a time‐index k ∈ {1,…,n_steps}
        if self.homogeneous_time:
            k = torch.randint(1, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k, device=device)
        else:
            k_idx = torch.randint(1, self.n_steps + 1, (B,), device=device)

        # 4) get corresponding t and jump‐fraction w_t
        t   = grid[k_idx]                                  # (B,)
        w_t = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1             # (B,)

        # 5) thinning N → N_t ~ Binomial(N, w_t)
        w_t = w_t.unsqueeze(-1).expand_as(N)               # broadcast to (B,d)
        N_t = torch.distributions.Binomial(N.float(), w_t).sample().long()

        # 6) hypergeometric draw for births up to t
        B_t_np = manual_hypergeometric(
            total_count = N.cpu().numpy(),
            num_successes = B1.cpu().numpy(),
            num_draws =     N_t.cpu().numpy()
        )
        B_t = torch.from_numpy(B_t_np).to(device).long()

        # 7) build signed X_t, then reflect
        X_t_signed = x0 + 2 * B_t - N_t                     # may be negative
        x_t = X_t_signed.abs()                              # reflected at 0

        return {
            "x0" : x0,   # start
            "x1" : x1,   # end
            "x_t": x_t,  # |X_t|
            "t"  : t,    # time
            "z"  : z,    # context
            "N"  : N,    # total jumps
            "B1" : B1,   # births at t=1
            "grid": grid,
            "Λp" : Λp,
            "Λm" : Λm,
        }

class PoissonMeanConstrainedBDBridgeCollate:
    """
    Same latent construction as PoissonBDBridgeCollate but the intermediate
    state x_t is *resampled* (via a fast MH swap) so that
            sum_i x_t^{(i)}  =  d * [ (1–t)·mean(x0) + t·mean(x1) ] .
    """

    def __init__(
        self,
        n_steps: int,
        lam_p0: float = 8.0,  lam_p1: float = 8.0,
        lam_m0: float = 8.0,  lam_m1: float = 8.0,
        schedule_type: str = "constant",
        time_schedule: str = "uniform",
        homogeneous_time: bool = True,
        mh_sweeps: int = 5,
        **schedule_kwargs,
    ):
        self.n_steps        = n_steps
        self.lam_p0, self.lam_p1 = lam_p0, lam_p1
        self.lam_m0, self.lam_m1 = lam_m0, lam_m1
        self.schedule_type  = schedule_type
        self.homogeneous_time = homogeneous_time
        self.time_points    = make_time_spacing_schedule(
                                  n_steps, time_schedule, **schedule_kwargs)
        self.mh_sweeps      = mh_sweeps
        self.sched_kwargs   = schedule_kwargs

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _sample_N_M_B1(diff, Λp1, Λm1, device):
        λ = 2.0 * torch.sqrt(Λp1 * Λm1)        # scalar
        λ = λ.unsqueeze(-1).expand_as(diff)   # (B,d)
        M  = torch.poisson(λ).long().to(device)
        N  = diff.abs() + 2 * M
        B1 = (N + diff) >> 1
        return N, M, B1

    # ------------------------------------------------------------------ main
    def __call__(self, batch):
        # unpack
        x0 = torch.stack([b["x0"] for b in batch])  # (B,d)
        x1 = torch.stack([b["x1"] for b in batch])
        z  = torch.stack([b["z"]  for b in batch])

        # dataloader might add an extra dim
        if x0.ndim == 3:
            x0, x1, z = x0.squeeze(0), x1.squeeze(0), z.squeeze(0)

        device = x0.device
        B, d   = x0.shape

        # λ–schedule
        grid, _, _, Λp, Λm = make_lambda_schedule(
            self.n_steps,
            self.lam_p0, self.lam_p1,
            self.lam_m0, self.lam_m1,
            self.schedule_type,
            device=device,
        )
        Λ_tot1 = Λp[-1] + Λm[-1]

        # latent (N, M, B1)
        diff = x1 - x0
        N, M, B1 = self._sample_N_M_B1(diff, Λp[-1], Λm[-1], device)

        # choose a random interior grid index k
        if self.homogeneous_time:
            k_idx_val = torch.randint(1, self.n_steps + 1, (1,), device=device).item()
            k_idx = torch.full((B,), k_idx_val, device=device)
        else:
            raise NotImplementedError("Non-homogeneous time is not implemented for mean-constrained bridge")

        t_val = grid[k_idx_val]
        t      = grid[k_idx]                            # (B,)
        w_t    = (Λp[k_idx] + Λm[k_idx]) / Λ_tot1       # (B,)
        N_t    = torch.binomial(N.float(), w_t.unsqueeze(-1)).long()  # (B,d)

        # births up to t
        B_t = torch.from_numpy(
                  manual_hypergeometric(N.cpu().numpy(),
                                        B1.cpu().numpy(),
                                        N_t.cpu().numpy())
              ).to(device)

        # raw x_t
        x_t_raw = x0 + 2 * B_t - N_t                   # (B,d)

        # ------------------------------------------------------------------
        #  Mean–constraint target  S_target
        # ------------------------------------------------------------------
        mu0  = x0.float().mean(dim=0)                  # (d,)
        mu1  = x1.float().mean(dim=0)                  # (d,)
        mu_t = (1. - t_val) * mu0 + t_val * mu1        # (d,)
        S_target = (mu_t * B).round().long()           # (B,)

        # expected x for proposal weights
        p_birth = B1.float() / N.float()
        E_Bt    = p_birth * N_t.float()
        x_exp   = x0.float() + 2*E_Bt - N_t.float()

        # MH projection
        x_t = mh_mean_constrained_update(
                  x      = x_t_raw,
                  x_exp  = x_exp,
                  S      = S_target,
                  a      = x0,               # here a = X₀, not x̂₀
                  N_t    = N_t,
                  B_t    = B_t,
                  N_s    = N_t,              # same time-slice
                  sweeps = self.mh_sweeps
              )

        # recompute diff & slack pairs after MH (keeps N_t fixed)
        diff_t = (x_t - x0)
        M_t    = ((N_t - diff_t.abs()) >> 1)

        return {
            "x0"   : x0,
            "x1"   : x1,
            "x_t"  : x_t,
            "t"    : t,          # (B,)
            "z"    : z,
            "N"    : N,
            "M"    : M_t,
            "B1"   : B1,
            "grid" : grid,
            "Λp"   : Λp,
            "Λm"   : Λm,
        }
