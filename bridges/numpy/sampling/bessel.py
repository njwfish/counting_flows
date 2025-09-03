import numpy as np
from scipy.special import iv, gammaln

def sample_bessel_devroye(alpha, beta, d, n_samples=None, rng=None):
    """
    Devroye’s exact O(1)-time rejection sampler for
    P[M=m | B1−D1=d] ∝ (αβ)^{m + d/2}/(m!(m+d)!) / I_d(2√(αβ)).
    """
    rng = np.random.default_rng() if rng is None else rng
    if isinstance(alpha, float):
        alpha = np.ones_like(d) * alpha
    if isinstance(beta, float):
        beta = np.ones_like(d) * beta

    λ = alpha * beta
    # 1) Locate the (unique) mode m0
    m0 = int(np.floor((np.sqrt(4*λ + d*d) - d) / 2))

    # 2) Compute p0 = P[M=m0] exactly (including λ^{d/2})
    a = 2.0 * np.sqrt(λ)
    log_p0 = (
        (m0 + 0.5*d) * np.log(λ)
        - (gammaln(m0+1) + gammaln(m0+d+1))
        - np.log(iv(d, a))
    )
    p0 = np.exp(log_p0)

    # 3) Envelope half‑width and mixture weight
    w = 1.0 + p0/2.0
    width = 2.0*w / p0

    # 4) Rejection loop
    shape = alpha.shape if n_samples is None else alpha.shape + (n_samples,)
    M = np.empty(n_samples, dtype=int)
    done = np.zeros(n_samples, dtype=bool)

    # where any log_p0 is inf or nan set M to 0 and done to True
    M[np.isinf(log_p0) | np.isnan(log_p0)] = 0
    done[np.isinf(log_p0) | np.isnan(log_p0)] = True

    # while not done.all():
    while not done.all():
        idx = ~done
        K = idx.sum()

        # mix: uniform‐box vs exponential‐tail
        U = rng.random(K)
        in_box = U < (w / (1.0 + w))

        # propose Y
        Y = np.empty(K)
        # uniform on [−w/p0, w/p0]
        if in_box.any():
            Y[in_box] = (rng.random(in_box.sum()) - 0.5) * width
        # tails: (w + Exp)/p0
        t = (~in_box).sum()
        if t:
            Y[~in_box] = (w + rng.exponential(size=t)) / p0

        # add random sign
        Y *= rng.choice([-1.0, 1.0], size=K)
        k_prop = np.rint(Y).astype(int)
        m_prop = m0 + k_prop

        # immediately reject negative indices
        invalid = (m_prop < 0)
        safe = np.where(invalid, 0, m_prop)

        # log‑ratio of pmfs (normalizer I_d cancels)
        log_ratio = (
            k_prop*np.log(λ)
            - (gammaln(safe+1) + gammaln(safe+d+1)
               - gammaln(m0+1) - gammaln(m0+d+1))
        )

        # correct acceptance: subtract max(0, p0*|Y| − w)
        delta = p0 * np.abs(Y) - w
        log_accept = log_ratio - np.maximum(0.0, delta)
        accept_prob = np.exp(log_accept)

        U2 = rng.random(K)
        accept = (~invalid) & (U2 < accept_prob)

        M[idx] = np.where(accept, m_prop, M[idx])
        done[idx] = accept

    return M
