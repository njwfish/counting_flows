import cupy as cp
from typing import Callable
from .scheduling import make_weight_schedule
from .sampling.mvhg import mvhg
from .sampling.ffh import ffh
from .sampling.prop import proportional_proj
from .utils import dlpack_backend

class SkellamMeanConstrainedBridge:
    """
    Same latent construction as SkellamBridge but the intermediate
    state x_t is *resampled* (via a fast MH swap) so that
            sum_i x_t^{(i)}  =  d * [ (1–t)·mean(x0) + t·mean(x1) ] .
    
    Numpy implementation - identical to PyTorch version.
    """

    def __init__(
        self,
        n_steps: int,
        m_sampler: Callable,
        schedule_type: str = "linear",
        mh_sweeps: int = 75,
        backend: str = "torch",
        **schedule_kwargs,
    ):
        self.n_steps        = n_steps
        self.m_sampler      = m_sampler
        self.schedule_type  = schedule_type
        self.time_points    = cp.linspace(0, 1, num=n_steps + 1)
        self.weights        = make_weight_schedule(
            n_steps, schedule_type, **schedule_kwargs
        )
        self.mh_sweeps      = mh_sweeps
        self.backend        = backend

    def __call__(self, x_0, x_1, t_target=None):
        # unpack
        x_0, x_1 = dlpack_backend(x_0, x_1, backend='cupy', dtype="int32")

        b, d   = x_0.shape

        # latent (N, M, B1)
        diff = x_1 - x_0
        M = self.m_sampler(diff)
        N_1 = cp.abs(diff) + 2 * M
        B_1 = (N_1 + diff) >> 1

        # choose time index - either specified or random
        if t_target is not None:
            # Find closest grid point to target time
            time_diffs = cp.abs(self.time_points - t_target)
            k = cp.argmin(time_diffs).item()
            k = cp.broadcast_to(k, (b,))
        else:
            k = cp.random.randint(0, self.n_steps + 1)
            k = cp.broadcast_to(k, (b,))

        t      = self.time_points[k].reshape(-1, 1)                            # (b,)
        w_t    = self.weights[k]                                # (b,)

        Nt_tot = (w_t.reshape(-1, 1) * N_1.sum(axis=0, keepdims=True)).round().astype(cp.int32)
        Bt_tot = (w_t.reshape(-1, 1) * B_1.sum(axis=0, keepdims=True)).round().astype(cp.int32)

        N_t = mvhg(pop=N_1.T, draws_tot=Nt_tot).T
        B_t = ffh(w=B_1.T, b=(N_1 - B_1).T, k=N_t.T, S=Bt_tot, sweeps=self.mh_sweeps).T

        x_t = x_1 - 2 * (B_1 - B_t) + (N_1 - N_t)

        # recompute diff & slack pairs after MH (keeps N_t fixed)
        diff_t = (x_t - x_0)
        M_t    = (N_t - cp.abs(diff_t)) >> 1

        return dlpack_backend(x_t, M_t, t, backend=self.backend, dtype="float32")

    def reverse_sampler(
        self,
        x_1: cp.ndarray,
        z:  dict,
        mu_0: cp.ndarray,
        model,
        return_trajectory: bool = False,
        return_x_hat:     bool = False,
        return_M:         bool = False,
        guidance_x0:      cp.ndarray = None,
        guidance_schedule: cp.ndarray = None,
    ):
        """
        Reverse sampler that *enforces the running batch mean*
            μ_k = (1−t_k)·μ₀ + t_k·μ₁
        at every grid point via an MH swap kernel.

        Identical to the plain sampler except that, after the BD step,
        the state is projected onto the hyper-plane
            { ‖x‖₁  =  round(B · μ_k) }.
        """
        b, d = x_1.shape
        x_1 = x_1.astype(cp.int32)
        if guidance_x0 is not None:
            guidance_x0 = guidance_x0.astype(cp.int32)

        mu_1 = x_1.mean(axis=0)
        S_0 = (mu_0 * b).round().astype(cp.int32)

        def sample_step(k, x_t, M_t=None, **z):
            if M_t is None:
                if not self.m_sampler.markov:
                    M_t = self.m_sampler(x_t)

            t = cp.broadcast_to(self.time_points[k], (b, 1))
            x_t_dl, M_t_dl, t_dl = dlpack_backend(x_t, M_t, t, backend=self.backend, dtype="float32")
            x0_hat_t = model.sample(x_t_dl, M_t_dl, t_dl, **z)
            x0_hat_t = dlpack_backend(x0_hat_t, backend='cupy', dtype="int32")

            if guidance_x_0 is not None:
                x0_hat_t =  guidance_schedule[k] * guidance_x_0 + (1 - guidance_schedule[k]) * x0_hat_t

            x0_hat_t = x0_hat_t.round().astype(cp.int32)
            x0_hat_t = proportional_proj(x0_hat_t.T, S_0).T
            diff = x_t - x0_hat_t

            # the markov sampler requires the diff to sample M_t
            if M_t is None:
                M_t = self.m_sampler(diff)
            N_t         = cp.abs(diff) + 2 * M_t
            B_t         = (N_t + diff) >> 1

            ρ = (self.weights[k-1] / self.weights[k])

            N_s_tot = (ρ * N_t.sum(axis=0)).round().astype(cp.int32)
            B_s_tot = (ρ * B_t.sum(axis=0)).round().astype(cp.int32)
            N_s = mvhg(pop=N_t.T, draws_tot=N_s_tot).T
            B_s = ffh(
                w=B_t.T, b=(N_t - B_t).T, k=N_s.T, S=B_s_tot, 
                sweeps=self.mh_sweeps
            ).T

            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s) 

            diff_s = cp.abs(x_s - x0_hat_t)
            M_s    = (N_s - diff_s) >> 1
            return x_s, M_s, x0_hat_t

        x_t, M_t, x0_hat_t = sample_step(self.n_steps, x_1, **z)

        traj, xhat_traj, M_traj = [x_t], [x0_hat_t], [M_t]
        for k in range(self.n_steps - 1, 0, -1):
            x_t, M_t, x0_hat_t = sample_step(k, x_t, M_t, **z)

            if return_trajectory: traj.append(x_t)
            if return_x_hat:      xhat_traj.append(x0_hat_t)
            if return_M:          M_traj.append(M_t)

        outs = [x_t]
        if return_trajectory: outs.append(cp.stack(traj))
        if return_x_hat:      outs.append(cp.stack(xhat_traj))
        if return_M:          outs.append(cp.stack(M_traj))
        return tuple(outs) if len(outs) > 1 else x_t 