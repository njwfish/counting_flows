import cupy as cp
from typing import Callable
from .scheduling import make_weight_schedule
from .utils import dlpack_backend

class SkellamBridge:
    """
    Collate for Skellam birth–death bridge with time‑varying λ₊, λ₋.
    Produces x_t drawn via Hypergeometric(N,B₁,N_t).
    """

    def __init__(
        self,
        n_steps: int,
        m_sampler: Callable,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        backend: str = "torch",
        **schedule_kwargs,
    ):
        self.n_steps          = n_steps
        self.m_sampler        = m_sampler
        self.schedule_type    = schedule_type
        self.time_points      = cp.linspace(0, 1, n_steps + 1)
        self.homogeneous_time = homogeneous_time
        self.weights          = make_weight_schedule(
            n_steps, schedule_type, **schedule_kwargs
        )
        self.backend          = backend

    def __call__(self, x_0, x_1, t_target=None):
        x_0, x_1 = dlpack_backend(x_0, x_1, backend='cupy', dtype="int32")

        b, d = x_0.shape

        diff = (x_1 - x_0)
        M = self.m_sampler(diff)
        N_1 = cp.abs(diff) + 2 * M
        B_1 = (N_1 + diff) // 2

        if t_target is not None:
            # find closest time point
            time_diffs = cp.abs(self.time_points - t_target)
            k = cp.argmin(time_diffs)
            k = cp.broadcast_to(k, (b,))

        elif self.homogeneous_time:
            k = cp.random.randint(1, self.n_steps + 1)
            k = cp.broadcast_to(k, (b,))
        else:
            k = cp.random.randint(1, self.n_steps + 1, (b,))
            
        t     = self.time_points[k].reshape(-1, 1)
        w_t   = self.weights[k]
        
        N_t = cp.random.binomial(N_1, cp.expand_dims(w_t, -1), dtype=cp.int32)
        
        # need to gaurd against zero nsample
        non_zero = N_t > 0
        B_t = cp.zeros_like(B_1)
        B_t[non_zero] = cp.random.hypergeometric(
            ngood=B_1[non_zero],
            nbad=N_1[non_zero] - B_1[non_zero],
            nsample=N_t[non_zero], 
            dtype=cp.int32
        )

        x_t = x_1 - 2 * (B_1 - B_t) + (N_1 - N_t)

        diff_t = cp.abs(x_t - x_0)
        M_t    = (N_t - diff_t) // 2

        return dlpack_backend(x_t, M_t, t, backend=self.backend, dtype="float32")
    
    def reverse_sampler(
        self,
        x_1:  cp.ndarray,
        z:   dict,
        model,
        return_trajectory: bool = False,
        return_x_hat:      bool = False,
        return_M:          bool = False,
        guidance_x_0:      cp.ndarray = None,
        guidance_schedule: cp.ndarray = None,
    ):
        b, d    = x_1.shape
        x_t     = cp.from_dlpack(x_1).round().astype(cp.int32)

        if guidance_x_0 is not None:
            guidance_x_0 = cp.from_dlpack(guidance_x_0).round().astype(cp.int32)

        def sample_step(k, x_t, M_t=None, **z):
            if M_t is None:
                if not self.m_sampler.markov:
                    M_t = self.m_sampler(x_t)

            print(k, self.time_points[k])
            t = cp.broadcast_to(self.time_points[k], (b, 1))
            x_t_dl, M_t_dl, t_dl = dlpack_backend(x_t, M_t, t, backend=self.backend, dtype="float32")
            x0_hat_t = model.sample(x_t_dl, M_t_dl, t_dl, **z)
            x0_hat_t = cp.from_dlpack(x0_hat_t)

            if guidance_x_0 is not None:
                x0_hat_t =  guidance_schedule[k] * guidance_x_0 + (1 - guidance_schedule[k]) * x0_hat_t

            x0_hat_t = x0_hat_t.round().astype(cp.int32)
            diff = x_t - x0_hat_t

            # the markov sampler requires the diff to sample M_t
            if M_t is None:
                M_t = self.m_sampler(diff)

            N_t         = cp.abs(diff) + 2 * M_t
            B_t         = (N_t + diff) // 2

            ρ = (self.weights[k-1] / self.weights[k])
            non_zero = N_t > 0
            N_s = cp.zeros_like(N_t)
            N_s[non_zero] = cp.random.binomial(N_t[non_zero], ρ)

            non_zero = N_s > 0
            B_s = cp.zeros_like(B_t)
            B_s[non_zero] = cp.random.hypergeometric(
                ngood   = B_t[non_zero],
                nbad    = N_t[non_zero] - B_t[non_zero],
                nsample = N_s[non_zero]
            )

            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)
            diff_s = cp.abs(x_s - x0_hat_t)
            M_s = (N_s - diff_s) // 2  

            return x_s, M_s, x0_hat_t
        
        M_t = self.m_sampler(x_1)
        x_t, M_t, x0_hat_t = sample_step(self.n_steps, x_t, M_t, **z)

        traj, xhat_traj, M_traj = [x_t], [x0_hat_t], [M_t]
        for k in range(self.n_steps - 1, 0, -1):
            
            x_t, M_t, x0_hat_t = sample_step(k, x_t, M_t, **z)

            if return_trajectory: traj.append(x_t)
            if return_x_hat:     xhat_traj.append(x0_hat_t)
            if return_M:         M_traj.append(M_t)

        outs = [x_t]
        if return_trajectory: outs.append(cp.stack(traj))
        if return_x_hat:      outs.append(cp.stack(xhat_traj))
        if return_M:          outs.append(cp.stack(M_traj))
        return tuple(outs) if len(outs) > 1 else x_t 