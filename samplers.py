"""
Reverse Sampling for Count-based Flow Matching

Provides the reverse sampling process that generates count data from noise
using trained neural networks and bridge kernels.
"""

import torch
from torch.distributions import Binomial, Beta, NegativeBinomial
from .scheduling import make_time_spacing_schedule, make_phi_schedule, get_bridge_parameters, make_lambda_schedule
from .bridges import manual_hypergeometric

from typing import Tuple

import torch
from torch.distributions import Multinomial

# -----------------------------------------------------------------------------
# 1) Sum‐constrained Multinomial proposal
# -----------------------------------------------------------------------------
def constrained_multinomial_proposal(x: torch.LongTensor,
                                     x_exp: torch.Tensor,
                                     S: torch.LongTensor):
    """
    Propose x' with sum=S by distributing the remainder R=S-sum(x)
    via a signed multinomial on weights ∝ max(sign*(x_exp - x),0).
    
    Args:
      x     (B,d) LongTensor   current counts
      x_exp (B,d) Tensor       expected counts E[x_i]
      S     (B,)   LongTensor  target sum per batch sample
    Returns:
      x_prop (B,d) LongTensor  proposed new counts (sum = S)
      sgn    (B,)   LongTensor  sign of the remainder
    """
    B,d = x.shape
    R    = S - x.sum(dim=1)           # (B,)
    sgn  = R.sign().long()            # in {-1,0,1}
    Rabs = R.abs().float()            # how many to allocate

    # directional weights
    diff    = sgn.unsqueeze(-1) * (x_exp - x).float()  # (B,d)
    weights = diff.clamp(min=0.0)                      # zero out opp. dir

    wsum      = weights.sum(dim=1, keepdim=True)       # (B,1)
    zero_mask = (wsum == 0)
    wsum     += zero_mask.float()                      # avoid div0
    probs     = weights / wsum                         # (B,d)

    # draw delta ~ Multinomial(Rabs, probs)
    m     = Multinomial(total_count=Rabs, probs=probs)
    delta = m.sample().round().long().to(x.device)     # (B,d)

    x_prop = x + sgn.unsqueeze(-1)*delta
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
    sweeps:  int = 5
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
        x_prop = x_prop.clamp(min=0)

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


# ------------------------------------------------------------------------------
#  Core BD step (unreflected)
# ------------------------------------------------------------------------------
import torch
from torch.distributions import Multinomial, Binomial

# --------------------------------------------------------------------
#  1) Core BD reverse‐kernel step (unreflected)
# --------------------------------------------------------------------
def bd_step(
    x_t:    torch.LongTensor,   # (B,d) current state at time t
    x0_hat: torch.LongTensor,   # (B,d) model prediction of X₀
    N_t:    torch.LongTensor,   # (B,d) latent total jumps at t
    B_t:    torch.LongTensor,   # (B,d) latent births at t
    w_t:    torch.Tensor,       # scalar or (B,)                = w(t)
    w_s:    torch.Tensor        # scalar or (B,)                = w(s)
) -> (torch.LongTensor, torch.LongTensor, torch.LongTensor):
    """
    One reverse step of the Birth–Death bridge (unreflected).
    Returns (x_s, N_s, B_s).
    """
    # 1) thinning: N_s ~ Binomial(N_t, w_s/w_t)
    p = (w_s / w_t).clamp(0.0, 1.0)
    if p.ndim == 1:
        p = p.unsqueeze(-1)         # (B,1)
    p = p.expand_as(N_t).float()    # (B,d)
    # use torch.distributions.Binomial for true Binomial
    binom = Binomial(total_count=N_t.float(), probs=p)
    N_s   = binom.sample().long()   # (B,d)

    # 2) births: B_s ~ Hypergeom(N_t, B_t, N_s)
    #    (replace with your fast vectorized version)
    B_s_np = manual_hypergeometric(
      total_count=N_t.cpu().numpy(),
      num_successes=B_t.cpu().numpy(),
      num_draws=N_s.cpu().numpy()
    )
    B_s = torch.from_numpy(B_s_np).to(N_t.device).long()

    # 3) reconstruct
    x_s = x0_hat + 2*B_s - N_s
    return x_s, N_s, B_s

# --------------------------------------------------------------------
#  Standard BD reverse‑time sampler WITH slack‑pair feature
# --------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
#  Required imports + helper
# ──────────────────────────────────────────────────────────────────────────────
import torch
from torch.distributions import Binomial

from .bridges import manual_hypergeometric


# ──────────────────────────────────────────────────────────────────────────────
#  Reverse–time sampler for the *standard* Birth–Death bridge
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def bd_reverse_sampler(
    x1:  torch.LongTensor,            # observed end‑point X₁  (B,d)
    z:   torch.Tensor,                # conditioning / context
    model,                            # nn predicting x̂₀(·, t, z, extra)
    K:         int,                   # #grid steps
    lam0:      float,                 # λ₊(0)=λ₋(0)
    lam1:      float,                 # λ₊(1)=λ₋(1)
    schedule_type:  str = "constant",
    time_schedule:  str = "uniform",
    return_trajectory: bool = False,
    return_x_hat:     bool = False,
    device: str = "cuda",
):
    """
    Exact reverse sampler with global slack‑pair counter M, thinned each step.
    At every grid time we pass M_t to the network.
    """
    device  = torch.device(device)
    x_t     = x1.to(device).long()        # current state (starts at X₁)
    B, d    = x_t.shape

    # 0.  Forward rate schedule and w(t)
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1, schedule_type, device=device
    )
    w = (Λp + Λm) / (Λp[-1] + Λm[-1])     # shape (K+1,)

    # 1.  Global slack‑pair count  M_global  (Poisson)
    λ_star   = 2.0 * torch.sqrt(Λp[-1] * Λm[-1])            # scalar tensor
    M_global = torch.poisson(λ_star).long().expand_as(x_t)  # (B,d)
    M_t      = M_global.clone()                             # at t = 1

    # 2.  First network prediction  (t = 1)
    t1        = torch.ones_like(x_t[:, :1])
    x0_hat_t  = model.sample(x_t, M_t, z, t1).round().long()
    diff      = x_t - x0_hat_t
    N_t       = diff.abs() + 2 * M_t                        # Eq. (N = |diff|+2M)
    B_t       = (N_t + diff) >> 1                           # births (integer)

    # optional logging
    traj, xhat_traj = [], []

    # 3.  Reverse sweep  t_k → t_{k-1}
    for k in range(K, 0, -1):
        t_val, s_val = grid[k], grid[k - 1]
        ρ            = (w[k - 1] / w[k]).item()             # scalar (0,1)

        # 3.a  Thin slack pairs:  M_s | M_t ~ Bin(M_t, ρ²)
        M_s = torch.distributions.Binomial(
                  total_count=M_t.float(), probs=ρ**2
        ).sample().long()

        # 3.b  Network prediction at *current* time (t_k)
        t_tensor = torch.full_like(x_t[:, :1], t_val)
        x0_hat_t = model.sample(x_t, M_t, z, t_tensor).round().long()
        diff     = x_t - x0_hat_t

        #      Re‑define N_t, B_t from diff and *current* M_t
        N_t = diff.abs() + 2 * M_t
        B_t = (N_t + diff) >> 1

        # 3.c  Binomial thinning of total jumps:  N_s | N_t
        N_s = torch.distributions.Binomial(
                  total_count=N_t.float(), probs=ρ
        ).sample().long()

        # 3.d  Hypergeometric thinning of births:  B_s | (N_t, B_t, N_s)
        B_s_np = manual_hypergeometric(
            total_count=N_t.cpu().numpy(),
            num_successes=B_t.cpu().numpy(),
            num_draws=N_s.cpu().numpy()
        )
        B_s = torch.from_numpy(B_s_np).to(device).long()


        # 3.f  Roll state for next iteration
        x_t -= 2 * (B_t - B_s) - (N_t - N_s)
        M_t = M_s 

        if return_trajectory:
            traj.append(x_t.cpu())
        if return_x_hat:
            xhat_traj.append(x0_hat_t.cpu())

    # 4.  Return
    if return_trajectory and return_x_hat:
        return x_t, traj, xhat_traj
    if return_trajectory:
        return x_t, traj
    if return_x_hat:
        return x_t, xhat_traj
    return x_t




# --------------------------------------------------------------------
#  3) Reflected BD reverse‐time sampler
# --------------------------------------------------------------------
# --------------------------------------------------------------------
#  helpers:  log‑choose, posterior sign weight, hypergeom sampler
# --------------------------------------------------------------------
import torch
from torch.distributions import Binomial
from .bridges import manual_hypergeometric          # you already have this

def _log_choose(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """log {n \choose k} with broadcasting."""
    return (
        torch.lgamma(n + 1) - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
    )

def _sample_sign_and_Bt(
    N_t:   torch.LongTensor,    # (B, d)
    x_abs: torch.LongTensor,    # (B, d)  |X_t|
    x0hat: torch.LongTensor,    # (B, d)  model â
    N0:    torch.LongTensor,    # (B, d)  total jumps N  (fixed)
    B0:    torch.LongTensor,    # (B, d)  births B₁      (fixed)
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Sample latent sign S_t ∈ {+1,‑1} and the compatible B_t.
    Implements Lemma 4.1 of the write‑up.
    """
    device = x_abs.device
    # candidate birth counts under each sign
    Bt_plus  = (N_t - (x_abs - x0hat)) // 2        # B_t^{(+)}
    Bt_minus = (N_t - (x_abs + x0hat)) // 2        # B_t^{(-)}

    # feasibility masks (counts must be integers in [0, N_t])
    ok_p  = (Bt_plus  >= 0) & (Bt_plus  <= N_t) & ((N_t + x_abs - x0hat) % 2 == 0)
    ok_m  = (Bt_minus >= 0) & (Bt_minus <= N_t) & ((N_t - x_abs - x0hat) % 2 == 0)

    # log‑weights  log π_+, log π_–
    minus_inf = torch.full_like(Bt_plus, -1e30, dtype=torch.float)
    logw_p = torch.where(
        ok_p,
        _log_choose(B0, Bt_plus) +
        _log_choose(N0 - B0, N_t - Bt_plus),
        minus_inf,
    )
    logw_m = torch.where(
        ok_m,
        _log_choose(B0, Bt_minus) +
        _log_choose(N0 - B0, N_t - Bt_minus),
        minus_inf,
    )

    #   S=+1  prob  ∝  exp(logw_p) ;  S=-1  prob ∝ exp(logw_m)
    max_logw = torch.max(torch.stack([logw_p, logw_m]), dim=0).values
    w_p = (logw_p - max_logw).exp()
    w_m = (logw_m - max_logw).exp()
    p_plus = w_p / (w_p + w_m + 1e-12)

    print("Bt_plus", Bt_plus)
    print("Bt_minus", Bt_minus)
    print("logw_p", logw_p)
    print("logw_m", logw_m)
    print("p_plus", p_plus)

    u = torch.rand_like(p_plus)
    S_t = torch.where(u < p_plus,  torch.ones_like(N_t), -torch.ones_like(N_t))

    # choose the compatible birth count
    B_t = torch.where(S_t == 1, Bt_plus, Bt_minus)
    return S_t.long(), B_t.long()

# --------------------------------------------------------------------
#  reflected BD step  (one grid tick t -> s)
# --------------------------------------------------------------------
def _reflected_bd_step(
    x_abs_t: torch.LongTensor,      # (B, d)  |X_t|
    S_t:     torch.LongTensor,      # (B, d)  latent signs at t
    x0hat:   torch.LongTensor,      # (B, d)  model â at t
    N_t:     torch.LongTensor,      # (B, d)
    N_tot:   torch.LongTensor,      # (B, d)  fixed total N
    B_tot:   torch.LongTensor,      # (B, d)  fixed births B₁
    w_t, w_s                       # scalars or (B,) tensors
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor,
           torch.LongTensor]:
    """
    One reverse reflected step.  Returns
        x_abs_s,  S_s,  N_s,  signed X_s  (needed by the model).
    """
    # 1) (re‑)sample sign S_t and compatible B_t
    S_t, B_t = _sample_sign_and_Bt(N_t, x_abs_t, x0hat, N_tot, B_tot)

    # 2) thinning: N_s ~ Binomial(N_t, w_s / w_t)
    p = (w_s / w_t).clamp(0.0, 1.0)
    if p.ndim == 1:                # broadcast to (B,d)
        p = p.unsqueeze(-1).expand_as(N_t).float()
    binom = Binomial(total_count=N_t.float(), probs=p)
    N_s   = binom.sample().long()

    # 3) births: B_s ~ Hypergeom(N_t, B_t, N_s)
    B_s_np = manual_hypergeometric(
        total_count=N_t.cpu().numpy(),
        num_successes=B_t.cpu().numpy(),
        num_draws=N_s.cpu().numpy(),
    )
    B_s = torch.from_numpy(B_s_np).to(N_t.device).long()

    # 4) update signed and absolute state
    X_signed_t = S_t * x_abs_t
    X_signed_s = X_signed_t - 2 * (B_t - B_s) - (N_t - N_s)
    x_abs_s    = X_signed_s.abs()
    S_s        = torch.sign(X_signed_s).long().clamp(min=0, max=1) * 2 - 1
    # (if X=0 we arbitrarily keep the previous sign)

    return x_abs_s, S_s, N_s, X_signed_s

# --------------------------------------------------------------------
#  reflected BD reverse sampler
# --------------------------------------------------------------------
@torch.no_grad()
def reflected_bd_reverse_sampler(
    x1_abs: torch.LongTensor,       # (B, d)  observed |X₁|
    z, model,
    K: int,
    lam0, lam1,
    schedule_type: str  = "constant",
    time_schedule:  str  = "uniform",
    return_trajectory: bool = False,
    return_x_hat:      bool = False,
    device:            str  = "cuda",
):
    """
    Reverse sampler that *observes only |X_t|* and tracks latent signs.

    Returns:
      x_abs_0                 (B,d)  final |X₀|
      (optional) trajectory   list[(B,d)]  backward path of |X_t|
      (optional) sign_traj    list[(B,d)]  latent sign history
    """
    device  = torch.device(device)
    x_abs_t = x1_abs.to(device).long()      # current |X_t|
    B, d    = x_abs_t.shape

    # ---- 1) rate schedule  ---------------------------------------------------
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam0, lam1, lam0, lam1,
        schedule_type, device=device
    )
    w = (Λp + Λm) / (Λp[-1] + Λm[-1])        # shape (K+1,)

    # ---- 2) initial latent sign, total N and B₁ ------------------------------
    # 2.1  uniform sign  S₁ ∈ {+1,‑1}
    S_t = torch.where(
        torch.rand_like(x_abs_t, dtype=torch.float) < 0.5,
        torch.ones_like(x_abs_t), -torch.ones_like(x_abs_t)
    ).long()
    X_signed = S_t * x_abs_t                 # (B,d)

    # 2.2  model’s â at t=1
    t1 = torch.ones((B,1), device=device)
    x0_hat = model.sample(X_signed, z, t1, use_mean=True).round().long()

    # 2.3  total jump counts  N  and births  B₁  (depend on X₁ sign)
    diff   = X_signed - x0_hat
    λ_star = 2.0 * torch.sqrt(Λp[-1] * Λm[-1])
    M      = torch.poisson(λ_star.unsqueeze(-1).expand_as(diff)).long()
    N_tot  = diff.abs() + 2 * M             # (B,d)
    B_tot  = (N_tot + diff) // 2            # births up to t=1
    N_t    = N_tot.clone()                  # current N_t  (starts at N)
    # (B_t is reconstructed on the fly each step)

    # ---- 3) storage ----------------------------------------------------------
    if return_trajectory:
        traj_abs  = []
    if return_x_hat:
        traj_x_hat = []

    # ---- 4) backward sweep ---------------------------------------------------
    for k in range(K, 0, -1):
        t_val = grid[k]
        t_tensor = torch.full((B,1), t_val, device=device)
        x0_hat = model.sample(X_signed, z, t_tensor, use_mean=True).round().long()

        x_abs_t, S_t, N_s, X_signed = _reflected_bd_step(
            x_abs_t   = x_abs_t,
            S_t       = S_t,
            x0hat     = x0_hat,
            N_t       = N_t,
            N_tot     = N_tot,
            B_tot     = B_tot,
            w_t       = w[k],
            w_s       = w[k-1],
        )
        # prepare for next tick
        N_t = N_s

        if return_trajectory:
            traj_abs.append(x_abs_t.cpu())
        if return_x_hat:
            traj_x_hat.append(x0_hat.cpu())

    # ---- 5) return -----------------------------------------------------------
    if return_trajectory and return_x_hat:
        return x_abs_t, traj_abs, traj_x_hat
    if return_trajectory:
        return x_abs_t, traj_abs
    if return_x_hat:
        return x_abs_t, traj_x_hat
    return x_abs_t



@torch.no_grad()
def bd_reverse_with_interpolated_mean(
    x1, z, model,
    K,
    lam_p0, lam_p1, lam_m0, lam_m1,
    mu0,                # scalar or (B,) tensor: desired mean at t=0
    sweeps=5,           # MH sweeps per time‐step
    schedule_type="constant",
    time_schedule="uniform",
    device="cuda"
):
    """
    BD reverse‐time sampler with per‐step sum‐constraints S_k computed by
    interpolating between mu1 = mean(x1) at t=1 and mu0 at t=0.
    Enforces sum(x_s)=round(mu_k * d) at each step via MH.
    """
    device = torch.device(device)
    x = x1.to(device).long()
    B, d = x.shape

    # compute mu1 per‐sample
    mu1 = x1.float().mean(dim=1).to(device)   # (B,)

    # normalize mu0 to a (B,) tensor
    mu0_t = torch.as_tensor(mu0, device=device)
    if mu0_t.ndim == 0:
        mu0_t = mu0_t.expand(B)

    # build λ schedule and w(t)
    grid, _, _, Λp, Λm = make_lambda_schedule(
        K, lam_p0, lam_p1, lam_m0, lam_m1,
        schedule_type, device=device
    )
    w = ((Λp + Λm) / (Λp[-1] + Λm[-1])).to(device)  # (K+1,)

    # sample latent (N_t,B_t) at t=1
    t1     = torch.ones((B,1), device=device)
    a1     = model.sample(x, z, t1, use_mean=True).round().long()
    diff   = x - a1
    λstar  = 2.0 * torch.sqrt(Λp[-1] * Λm[-1])
    M      = torch.poisson(λstar.unsqueeze(-1).expand_as(diff)).long()
    N_t    = diff.abs() + 2*M
    B_t    = (N_t + diff)//2

    # backward sweep
    for k in range(K, 0, -1):
        # time points
        t = grid[k]
        s = grid[k-1]

        # model prediction at t_k
        t_tensor = torch.full((B,1), t, device=device)
        a = model.sample(x, z, t_tensor, use_mean=True).round().long()

        # 1) raw BD reverse‐step
        x0, N_s, B_s = bd_step(
            x_t=x, x0_hat=a,
            N_t=N_t, B_t=B_t,
            w_t=w[k], w_s=w[k-1]
        )

        # 2) compute hypergeom‐mean for proposal
        E_Bs = B_t.float().div(N_t.float().clamp(min=1e-6)) * N_s.float()
        x_exp = a.float() + 2*E_Bs - N_s.float()

        # 3) compute interpolated target mean μ_s
        mu_s = (1.0 - s)*mu0_t + s*mu1           # (B,)
        S_k  = (mu_s * d).round().long()         # (B,)

        # 4) MH to enforce sum(x0)=S_k
        x = mh_mean_constrained_update(
            x      = x0,
            x_exp  = x_exp,
            S      = S_k,
            a      = a,
            N_t    = N_t,
            B_t    = B_t,
            N_s    = N_s,
            sweeps = sweeps
        )

        # 5) update latents for next iteration
        N_t = N_s
        B_t = ((N_s + (x - a)) >> 1)

    return x
