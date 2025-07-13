import torch
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all
from torch.distributions.constraints import Constraint, integer_interval
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from ..utils import maybe_compile
# Enable JIT compilation for better performance


def hypergeometric_torch(total_count, success_count, num_draws):
    """
    Internal torch-based hypergeometric sampling function.
    Assumes all inputs are already torch tensors on the same device.
    """
    success_count = success_count.long()
    num_draws = num_draws.long()
    total_count = total_count.long()
    
    # Handle edge cases with clipping
    success_count = torch.clamp(success_count, min=0)
    success_count = torch.min(success_count, total_count)
    num_draws = torch.clamp(num_draws, min=0)
    num_draws = torch.min(num_draws, total_count)
    
    # Handle zero cases
    zero_draws = (num_draws == 0)
    zero_success = (success_count == 0)
    all_draws = (num_draws >= total_count)
    all_success = (success_count >= total_count)
    
    # Initialize result
    result = torch.zeros_like(total_count, dtype=torch.int64)
    
    # Handle edge cases first
    result[zero_draws] = 0
    result[zero_success] = 0
    result[all_draws] = success_count[all_draws]  
    result[all_success] = num_draws[all_success]
    
    # Find cases that need actual sampling
    need_sampling = ~(zero_draws | zero_success | all_draws | all_success)
    
    if need_sampling.any():
        # Sample using our Hypergeometric distribution
        dist = Hypergeometric(
            total_count=total_count[need_sampling],
            success_count=success_count[need_sampling], 
            num_draws=num_draws[need_sampling],
            validate_args=False  # We already handled edge cases
        )
        result[need_sampling] = dist.sample()

    return result


def hypergeometric_numpy(total_count, success_count, num_draws):
    """
    Internal numpy-based hypergeometric sampling function.
    Assumes all inputs are already numpy arrays with compatible shapes.
    """
    # Handle edge cases with clipping
    success_count = np.clip(success_count, 0, None)
    success_count = np.minimum(success_count, total_count)
    num_draws = np.clip(num_draws, 0, None)
    num_draws = np.minimum(num_draws, total_count)
    
    # Handle edge cases manually for consistency with torch version
    zero_draws = (num_draws == 0)
    zero_success = (success_count == 0)
    all_draws = (num_draws >= total_count)
    all_success = (success_count >= total_count)
    
    # Initialize result
    result = np.zeros_like(total_count)
    
    # Handle edge cases
    result[zero_draws] = 0
    result[zero_success] = 0
    result[all_draws] = success_count[all_draws]
    result[all_success] = num_draws[all_success]
    
    # Find cases that need actual sampling
    need_sampling = ~(zero_draws | zero_success | all_draws | all_success)
    
    if need_sampling.any():
        # Use numpy hypergeometric sampler for cases that need sampling
        result[need_sampling] = np.random.hypergeometric(
            success_count[need_sampling], 
            total_count[need_sampling] - success_count[need_sampling], 
            num_draws[need_sampling]
        )
    
    return result


@torch.no_grad()
def hypergeometric(total_count, success_count, num_draws, backend='auto'):
    """
    Pure PyTorch vectorized hypergeometric sampling function with smart backend selection.
    
    Drop-in replacement for manual_hypergeometric that:
    1. Accepts torch tensors, numpy arrays, or scalars
    2. Returns same type/device as inputs 
    3. Uses numpy sampler for numpy inputs, torch sampler for torch inputs by default
    4. Allows override to force specific backend
    5. Handles conversions efficiently to minimize overhead
    
    Args:
        total_count: Total population size (tensor, array, or scalar)
        success_count: Number of success items in population (tensor, array, or scalar)
        num_draws: Number of items drawn (tensor, array, or scalar)
        backend: 'auto' (default), 'numpy', or 'torch' to control which sampler to use
        
    Returns:
        Same type as inputs: Number of successes in drawn samples (same shape as broadcasted inputs)
    """
    
    # Determine input types and original format info
    input_types = []
    devices = []
    original_inputs = [total_count, success_count, num_draws]
    
    for inp in original_inputs:
        if isinstance(inp, torch.Tensor):
            input_types.append('torch')
            devices.append(inp.device)
        elif isinstance(inp, np.ndarray):
            input_types.append('numpy')
            devices.append(None)
        else:
            input_types.append('scalar')
            devices.append(None)
    
    # Determine the primary input type and device
    if 'torch' in input_types:
        primary_type = 'torch'
        # Find a GPU device if any input is on GPU, otherwise use first torch device
        primary_device = None
        for device in devices:
            if device is not None:
                if primary_device is None or device.type == 'cuda':
                    primary_device = device
        if primary_device is None:
            primary_device = torch.device('cpu')
    elif 'numpy' in input_types:
        primary_type = 'numpy' 
        primary_device = None
    else:
        primary_type = 'scalar'
        primary_device = None
    
    # Decide which backend to use
    if backend == 'auto':
        use_backend = primary_type if primary_type in ['numpy', 'torch'] else 'torch'
    elif backend in ['numpy', 'torch']:
        use_backend = backend
    else:
        raise ValueError(f"backend must be 'auto', 'numpy', or 'torch', got {backend}")
    
    # Convert inputs and dispatch to appropriate backend
    if use_backend == 'numpy':
        # Convert everything to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.array(x)
        
        total_count_np = to_numpy(total_count)
        success_count_np = to_numpy(success_count)
        num_draws_np = to_numpy(num_draws)
        
        # Ensure integer types and handle broadcasting
        total_count_np = np.asarray(total_count_np, dtype=np.int64)
        success_count_np = np.asarray(success_count_np, dtype=np.int64) 
        num_draws_np = np.asarray(num_draws_np, dtype=np.int64)
        
        # Broadcast to common shape
        total_count_np, success_count_np, num_draws_np = np.broadcast_arrays(
            total_count_np, success_count_np, num_draws_np)
        
        # Call numpy backend
        result_np = hypergeometric_numpy(total_count_np, success_count_np, num_draws_np)
        
        # Convert result back to original format
        if primary_type == 'torch':
            result = torch.from_numpy(result_np).to(primary_device)
        elif primary_type == 'numpy':
            result = result_np
        else:  # scalar
            if result_np.ndim == 0:
                result = result_np.item()
            else:
                result = result_np
            
    else:  # use_backend == 'torch'
        # Convert everything to torch
        if isinstance(total_count, np.ndarray):
            total_count = torch.from_numpy(total_count)
        elif not isinstance(total_count, torch.Tensor):
            total_count = torch.tensor(total_count)
            
        if isinstance(success_count, np.ndarray):
            success_count = torch.from_numpy(success_count)
        elif not isinstance(success_count, torch.Tensor):
            success_count = torch.tensor(success_count)
            
        if isinstance(num_draws, np.ndarray):
            num_draws = torch.from_numpy(num_draws)
        elif not isinstance(num_draws, torch.Tensor):
            num_draws = torch.tensor(num_draws)
        
        # Ensure integer types
        total_count = total_count.long()
        success_count = success_count.long()
        num_draws = num_draws.long()
        
        # Get the device (prioritize GPU if any input is on GPU)
        device = total_count.device
        for tensor in [success_count, num_draws]:
            if tensor.device.type == 'cuda':
                device = tensor.device
                break
        
        # Move all to the same device
        total_count = total_count.to(device)
        success_count = success_count.to(device)
        num_draws = num_draws.to(device)
        
        # Call torch backend
        result_torch = hypergeometric_torch(total_count, success_count, num_draws)
        
        # Convert result back to original format if needed
        if primary_type == 'numpy':
            result = result_torch.cpu().numpy()
        elif primary_type == 'torch':
            result = result_torch
        else:  # scalar
            if result_torch.numel() == 1:
                result = result_torch.item()
            else:
                result = result_torch.cpu().numpy()
    
    return result



class Hypergeometric(Distribution):
    """
    Hypergeometric distribution implemented in pure PyTorch.
    
    The hypergeometric distribution models drawing without replacement from a 
    finite population containing exactly `total_count` objects, of which 
    `success_count` are successes. We draw `num_draws` objects.
    
    Args:
        total_count (int or Tensor): population size
        success_count (int or Tensor): number of success objects in population  
        num_draws (int or Tensor): number of draws
    """
    
    arg_constraints = {
        'total_count': integer_interval(0, float('inf')),
        'success_count': integer_interval(0, float('inf')), 
        'num_draws': integer_interval(0, float('inf'))
    }
    
    def __init__(self, total_count, success_count, num_draws, validate_args=None):
        self.total_count = torch.as_tensor(total_count, dtype=torch.long)
        self.success_count = torch.as_tensor(success_count, dtype=torch.long) 
        self.num_draws = torch.as_tensor(num_draws, dtype=torch.long)
        
        self.total_count, self.success_count, self.num_draws = broadcast_all(
            self.total_count, self.success_count, self.num_draws
        )
        
        batch_shape = self.total_count.shape
        super().__init__(batch_shape, validate_args=validate_args)
        
        if validate_args is not False:
            self._validate_args()
    
    def _validate_args(self):
        """Validate distribution parameters."""
        if not (self.total_count >= 0).all():
            raise ValueError("total_count must be non-negative")
        if not (self.success_count >= 0).all():
            raise ValueError("success_count must be non-negative") 
        if not (self.num_draws >= 0).all():
            raise ValueError("num_draws must be non-negative")
        if not (self.success_count <= self.total_count).all():
            raise ValueError("success_count must be <= total_count")
        if not (self.num_draws <= self.total_count).all():
            raise ValueError("num_draws must be <= total_count")
    
    @property
    def mean(self):
        """Mean of the hypergeometric distribution."""
        return self.num_draws * self.success_count.float() / self.total_count.float()
    
    @property
    def variance(self):
        """Variance of the hypergeometric distribution."""
        p = self.success_count.float() / self.total_count.float()
        n = self.num_draws.float()
        N = self.total_count.float()
        return n * p * (1 - p) * (N - n) / (N - 1)
    
    @property
    def support(self):
        """Support of the hypergeometric distribution."""
        lower = torch.max(torch.zeros_like(self.num_draws), 
                         self.num_draws - (self.total_count - self.success_count))
        upper = torch.min(self.num_draws, self.success_count)
        return (lower, upper)
    
    def sample(self, sample_shape=torch.Size()):
        """Sample from the hypergeometric distribution with optimized method selection."""
        if sample_shape:
            extended_shape = sample_shape + self.batch_shape
        else:
            extended_shape = self.batch_shape
            
        # Expand parameters to the full sample shape
        total_count = self.total_count.expand(extended_shape)
        success_count = self.success_count.expand(extended_shape) 
        num_draws = self.num_draws.expand(extended_shape)
        
        # Simplified method selection - just simple vs HRUA
        use_hrua = (num_draws >= 5) & (num_draws <= total_count - 5)
        
        # Initialize result tensor
        result = torch.zeros_like(total_count)
        
        # Simple sampling method for small samples
        simple_mask = ~use_hrua
        if simple_mask.any():
            result[simple_mask] = self._sample_simple_fast(
                total_count[simple_mask],
                success_count[simple_mask], 
                num_draws[simple_mask]
            )
        
        # HRUA method for larger samples
        if use_hrua.any():
            hrua_mask = use_hrua
            result[hrua_mask] = self._sample_hrua_fast(
                total_count[hrua_mask],
                success_count[hrua_mask],
                num_draws[hrua_mask]
            )
            
        return result
    
    @maybe_compile
    def _sample_simple_fast(self, total_count, success_count, num_draws):
        """Ultra-fast simple sampling using pure tensor operations."""
        device = total_count.device
        
        # Handle edge case where num_draws = 0 first (must return 0)
        zero_draws = num_draws == 0
        if zero_draws.any():
            result = torch.zeros_like(num_draws)
            # If all samples have zero draws, return immediately
            if zero_draws.all():
                return result
            # Continue processing non-zero cases below, will combine at the end
        
        # Handle case where sample > total/2 (vectorized)
        flip = num_draws > total_count // 2
        computed_sample = torch.where(flip, total_count - num_draws, num_draws)
        
        # Fast path for other edge cases (vectorized)  
        no_sample = computed_sample == 0  # This should now only be from flip logic
        all_good = total_count == success_count  
        no_good = success_count == 0
        draw_all = num_draws == total_count  # Fix: handle drawing everything
        
        # For edge cases, return immediately
        edge_case = no_sample | no_good
        if edge_case.any():
            # Handle each edge case properly
            result_edge = torch.zeros_like(computed_sample)
            # If drawing all items, result is success_count (not computed_sample)
            result_edge = torch.where(draw_all, success_count, result_edge)
            # If all items are successes, take min(computed_sample, success_count)
            result_edge = torch.where(all_good & ~draw_all, torch.min(computed_sample, success_count), result_edge)
            # Apply flip logic
            final_result_edge = torch.where(flip, result_edge, success_count - result_edge)
            # But if draw_all, result should just be success_count regardless of flip
            final_result_edge = torch.where(draw_all, success_count, final_result_edge)
            
            # Combine with zero_draws case if needed
            if 'zero_draws' in locals() and zero_draws.any():
                final_result_edge = torch.where(zero_draws, torch.zeros_like(final_result_edge), final_result_edge)
            
            return final_result_edge
        
        # For edge cases that can be resolved immediately, do so
        if edge_case.all():
            result_simple = torch.zeros_like(computed_sample)
            result_simple = torch.where(all_good, computed_sample, result_simple)
            final_simple = torch.where(flip, result_simple, success_count - result_simple)
            
            # Combine with zero_draws case if needed
            if 'zero_draws' in locals() and zero_draws.any():
                final_simple = torch.where(zero_draws, torch.zeros_like(final_simple), final_simple)
                
            return final_simple
        
        # Main sampling using vectorized approach
        remaining_total = total_count.clone().float()
        remaining_good = success_count.clone().float()
        remaining_sample = computed_sample.clone()
        
        # Pre-allocate maximum possible random numbers
        max_iterations = computed_sample.max()
        if max_iterations > 0:
            # Generate all random numbers at once for better performance
            batch_size = total_count.numel()
            all_randoms = torch.rand((max_iterations, batch_size), device=device)
            
            for i in range(max_iterations):
                # Active samples mask
                active = (remaining_sample > 0) & (remaining_good > 0) & (remaining_total > remaining_good)
                if not active.any():
                    break
                
                # Update remaining_total
                remaining_total = torch.where(active, remaining_total - 1, remaining_total)
                
                # Vectorized random decision
                rand_vals = all_randoms[i] * (remaining_total + 1)
                selected_good = (rand_vals < remaining_good) & active
                
                # Update counters
                remaining_good = torch.where(selected_good, remaining_good - 1, remaining_good)
                remaining_sample = torch.where(active, remaining_sample - 1, remaining_sample)
        
        # Handle final case where only good choices left
        only_good_left = remaining_total == remaining_good
        remaining_good = torch.where(only_good_left, remaining_good - remaining_sample, remaining_good)
        
        # Calculate final result
        final_result = torch.where(flip, remaining_good.long(), success_count - remaining_good.long())
        
        # Handle zero_draws case
        if 'zero_draws' in locals() and zero_draws.any():
            final_result = torch.where(zero_draws, torch.zeros_like(final_result), final_result)
            
        return final_result
    
    @maybe_compile
    def _sample_hrua_fast(self, total_count, success_count, num_draws):
        """Ultra-fast HRUA sampling without .item() calls for better compilation."""
        device = total_count.device
        dtype = torch.float32
        
        # Vectorized parameter computation
        computed_sample = torch.min(num_draws, total_count - num_draws)
        mingoodbad = torch.min(success_count, total_count - success_count)
        maxgoodbad = torch.max(success_count, total_count - success_count)
        
        p = mingoodbad.float() / total_count.float()
        q = maxgoodbad.float() / total_count.float()
        
        mu = computed_sample.float() * p
        a = mu + 0.5
        
        var = ((total_count - computed_sample).float() * computed_sample.float() * 
               p * q / (total_count.float() - 1))
        c = torch.sqrt(var + 0.5)
        
        # Constants
        D1 = 1.7155277699214135
        D2 = 0.8989161620588988
        h = D1 * c + D2
        
        # Mode calculation
        m = torch.floor((computed_sample.float() + 1) * (mingoodbad.float() + 1) / 
                       (total_count.float() + 2)).long()
        
        # Precompute log factorials at mode
        g = (torch.lgamma(m.float() + 1) + 
             torch.lgamma((mingoodbad - m).float() + 1) +
             torch.lgamma((computed_sample - m).float() + 1) + 
             torch.lgamma((maxgoodbad - computed_sample + m).float() + 1))
        
        b = torch.min(torch.min(computed_sample, mingoodbad) + 1,
                     torch.floor(a + 16 * c).long())
        
        # Vectorized rejection sampling
        K = torch.zeros_like(computed_sample)
        remaining_mask = torch.ones_like(computed_sample, dtype=torch.bool)
        
        # Fixed number of attempts to avoid dynamic shapes
        max_attempts = 500  # Reduced but should be sufficient
        batch_size = total_count.numel()
        
        if batch_size > 0:
            # Pre-generate all random numbers to avoid repeated calls
            U_all = torch.rand((max_attempts, batch_size), device=device, dtype=dtype)
            V_all = torch.rand((max_attempts, batch_size), device=device, dtype=dtype)
            
            for attempt in range(max_attempts):
                # Current mask for remaining samples
                current_remaining = remaining_mask.sum()
                if current_remaining == 0:
                    break
                
                # Get random numbers for this attempt (for all samples, mask later)
                U = U_all[attempt]
                V = V_all[attempt]
                
                # Get parameters for all samples (will mask later)
                X = a + h * (V - 0.5) / U
                valid_X = (X >= 0.0) & (X < b.float()) & remaining_mask
                
                if valid_X.any():
                    K_candidate = torch.floor(X).long()
                    
                    # Calculate acceptance probability (vectorized)
                    gp = (torch.lgamma(K_candidate.float() + 1) +
                          torch.lgamma((mingoodbad - K_candidate).float() + 1) +
                          torch.lgamma((computed_sample - K_candidate).float() + 1) +
                          torch.lgamma((maxgoodbad - computed_sample + K_candidate).float() + 1))
                    
                    T = g - gp
                    
                    # Acceptance tests (vectorized)
                    accept1 = (U * (4.0 - U) - 3.0) <= T
                    reject2 = ~accept1 & (U * (U - T) >= 1)
                    accept2 = ~accept1 & ~reject2 & (2.0 * torch.log(U) <= T)
                    
                    final_accept = (accept1 | accept2) & valid_X
                    
                    if final_accept.any():
                        # Update results for accepted samples
                        K = torch.where(final_accept, K_candidate, K)
                        remaining_mask = remaining_mask & ~final_accept
        
        # Handle flipped cases (vectorized)
        flip_good_bad = success_count > (total_count - success_count)
        K = torch.where(flip_good_bad, computed_sample - K, K)
        
        flip_sample = computed_sample < num_draws
        K = torch.where(flip_sample, success_count - K, K)
        
        return K
    
    def _log_factorial(self, n):
        """Compute log factorial using lgamma."""
        return torch.lgamma(n.float() + 1)
    
    def log_prob(self, value):
        """Compute log probability mass function."""
        value = torch.as_tensor(value, dtype=torch.long)
        
        # Check support
        lower, upper = self.support
        if not ((value >= lower) & (value <= upper)).all():
            # Return -inf for values outside support
            result = torch.full_like(value.float(), float('-inf'))
            valid_mask = (value >= lower) & (value <= upper)
            if valid_mask.any():
                result[valid_mask] = self._log_prob_valid(value[valid_mask])
            return result
        
        return self._log_prob_valid(value)
    
    def _log_prob_valid(self, value):
        """Compute log PMF for valid values."""
        # log P(X = k) = log(C(K,k) * C(N-K,n-k) / C(N,n))
        # Where C(n,k) is binomial coefficient
        
        K = self.success_count.float()
        N = self.total_count.float() 
        n = self.num_draws.float()
        k = value.float()
        
        # Use log binomial coefficients
        log_C_K_k = self._log_binomial_coeff(K, k)
        log_C_NK_nk = self._log_binomial_coeff(N - K, n - k)
        log_C_N_n = self._log_binomial_coeff(N, n)
        
        return log_C_K_k + log_C_NK_nk - log_C_N_n
    
    def _log_binomial_coeff(self, n, k):
        """Compute log binomial coefficient log(C(n,k)) with improved numerical stability."""
        # Handle edge cases first
        n = n.float()
        k = k.float()
        
        # Check for invalid inputs
        invalid_mask = (k < 0) | (k > n) | (n < 0)
        
        # For better numerical stability, use the fact that C(n,k) = C(n, n-k)
        # and always compute the smaller of the two
        k = torch.min(k, n - k)
        
        # Handle edge cases
        zero_mask = (k == 0)
        
        # Use double precision for intermediate calculations to reduce precision loss
        n_double = n.double()
        k_double = k.double()
        
        # Compute using lgamma with double precision
        result = (torch.lgamma(n_double + 1) - 
                 torch.lgamma(k_double + 1) - 
                 torch.lgamma(n_double - k_double + 1))
        
        # Convert back to float32
        result = result.float()
        
        # Handle edge cases explicitly
        result = torch.where(zero_mask, torch.zeros_like(result), result)
        result = torch.where(invalid_mask, torch.full_like(result, float('-inf')), result)
        
        return result

    def to(self, device):
        """Move distribution to specified device (CPU or GPU)."""
        self.total_count = self.total_count.to(device)
        self.success_count = self.success_count.to(device)
        self.num_draws = self.num_draws.to(device)
        return self
        
    @property
    def device(self):
        """Get the device of the distribution parameters."""
        return self.total_count.device


def test_hypergeometric():
    """Comprehensive test suite for the hypergeometric distribution."""
    print("Testing Hypergeometric Distribution Implementation")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    dist = Hypergeometric(total_count=20, success_count=7, num_draws=12)
    samples = dist.sample((1000,))
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min().item()}, {samples.max().item()}]")
    print(f"Expected range: [max(0, 12-13), min(12, 7)] = [0, 7]")
    
    # Test 2: Vectorized operations
    print("\n2. Testing vectorized operations...")
    total_counts = torch.tensor([20, 30, 15])
    success_counts = torch.tensor([7, 10, 5]) 
    num_draws = torch.tensor([12, 15, 8])
    
    dist_vec = Hypergeometric(total_counts, success_counts, num_draws)
    samples_vec = dist_vec.sample((100,))
    print(f"Vectorized sample shape: {samples_vec.shape}")
    
    # Test 3: Log probability mass function
    print("\n3. Testing log PMF...")
    test_values = torch.arange(0, 8)
    log_probs = dist.log_prob(test_values)
    probs = torch.exp(log_probs)
    print(f"PMF values sum: {probs.sum().item():.6f} (should be ~1.0)")
    
    # Test 4: Statistical properties
    print("\n4. Testing statistical properties...")
    large_sample = dist.sample((10000,))
    empirical_mean = large_sample.float().mean()
    theoretical_mean = dist.mean
    empirical_var = large_sample.float().var()
    theoretical_var = dist.variance
    
    print(f"Empirical mean: {empirical_mean.item():.4f}")
    print(f"Theoretical mean: {theoretical_mean.item():.4f}")
    print(f"Mean difference: {abs(empirical_mean - theoretical_mean).item():.4f}")
    
    print(f"Empirical variance: {empirical_var.item():.4f}")
    print(f"Theoretical variance: {theoretical_var.item():.4f}")
    print(f"Variance difference: {abs(empirical_var - theoretical_var).item():.4f}")
    
    # Test 5: Edge cases
    print("\n5. Testing edge cases...")
    
    # All draws are successes
    dist_edge1 = Hypergeometric(total_count=10, success_count=10, num_draws=5)
    samples_edge1 = dist_edge1.sample((100,))
    print(f"All successes case - samples should all be 5: {torch.unique(samples_edge1)}")
    
    # No successes available
    dist_edge2 = Hypergeometric(total_count=10, success_count=0, num_draws=5) 
    samples_edge2 = dist_edge2.sample((100,))
    print(f"No successes case - samples should all be 0: {torch.unique(samples_edge2)}")
    
    # Draw all items
    dist_edge3 = Hypergeometric(total_count=5, success_count=3, num_draws=5)
    samples_edge3 = dist_edge3.sample((100,))
    print(f"Draw all case - samples should all be 3: {torch.unique(samples_edge3)}")
    
    # Test 6: Probability correctness
    print("\n6. Testing probability correctness...")
    # Compare with known values for small examples
    dist_small = Hypergeometric(total_count=5, success_count=3, num_draws=2)
    
    # Manual calculation for verification:
    # P(X=0) = C(3,0)*C(2,2)/C(5,2) = 1*1/10 = 0.1
    # P(X=1) = C(3,1)*C(2,1)/C(5,2) = 3*2/10 = 0.6  
    # P(X=2) = C(3,2)*C(2,0)/C(5,2) = 3*1/10 = 0.3
    
    expected_probs = torch.tensor([0.1, 0.6, 0.3])
    computed_probs = torch.exp(dist_small.log_prob(torch.tensor([0, 1, 2])))
    
    print(f"Expected probabilities: {expected_probs}")
    print(f"Computed probabilities: {computed_probs}")
    print(f"Max difference: {(expected_probs - computed_probs).abs().max().item():.6f}")
    
    # Test 7: Large scale performance
    print("\n7. Testing large scale performance...")
    dist_large = Hypergeometric(total_count=1000, success_count=300, num_draws=100)
    
    import time
    start_time = time.time()
    large_samples = dist_large.sample((10000,))
    sampling_time = time.time() - start_time
    
    start_time = time.time()
    test_vals = torch.arange(50, 71)  # Around the mean
    large_log_probs = dist_large.log_prob(test_vals)
    logprob_time = time.time() - start_time
    
    print(f"Large scale sampling time (10k samples): {sampling_time:.4f}s")
    print(f"Large scale log_prob time (21 values): {logprob_time:.4f}s")
    print(f"Large sample mean: {large_samples.float().mean().item():.2f}")
    print(f"Large sample std: {large_samples.float().std().item():.2f}")
    
    # Test 8: Batch processing
    print("\n8. Testing batch processing...")
    batch_total = torch.randint(50, 200, (5, 3))
    batch_success = torch.randint(10, batch_total.min().item(), (5, 3))
    batch_draws = torch.randint(5, 30, (5, 3))
    
    dist_batch = Hypergeometric(batch_total, batch_success, batch_draws)
    batch_samples = dist_batch.sample((10,))
    
    print(f"Batch sample shape: {batch_samples.shape}")
    print(f"Batch parameters shape: {dist_batch.batch_shape}")
    
    # Test 9: Gradient compatibility
    print("\n9. Testing gradient compatibility...")
    # Test that log_prob works with autograd
    total_count_grad = torch.tensor(20.0, requires_grad=True)
    success_count_grad = torch.tensor(7.0, requires_grad=True)
    num_draws_grad = torch.tensor(12.0, requires_grad=True)
    
    # Create distribution (note: discrete distributions typically don't have gradients
    # with respect to parameters, but we can test the computational graph)
    test_val = torch.tensor(4.0)
    
    # Direct computation for gradient test
    log_prob_manual = (torch.lgamma(success_count_grad + 1) - 
                      torch.lgamma(test_val + 1) - 
                      torch.lgamma(success_count_grad - test_val + 1) +
                      torch.lgamma(total_count_grad - success_count_grad + 1) -
                      torch.lgamma(num_draws_grad - test_val + 1) -
                      torch.lgamma(total_count_grad - success_count_grad - num_draws_grad + test_val + 1) -
                      torch.lgamma(total_count_grad + 1) +
                      torch.lgamma(num_draws_grad + 1) +
                      torch.lgamma(total_count_grad - num_draws_grad + 1))
    
    log_prob_manual.backward()
    print(f"Gradient test passed - computation graph works")
    
    # Test 10: Visual verification with plots
    print("\n10. Testing with visual verification...")
    test_with_plots()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")


def test_with_plots():
    """Test sampling accuracy with visual verification across different regimes."""
    
    # Test configurations: small regime, large regime, and edge cases
    test_configs = [
        # (total_count, success_count, num_draws, title, regime)
        (20, 7, 5, "Small Regime: Simple Sampling", "small"),
        (20, 7, 12, "Medium Regime: Simple Sampling", "small"), 
        (100, 30, 25, "Large Regime: HRUA Sampling", "large"),
        (1000, 300, 100, "Very Large Regime: HRUA Sampling", "large"),
        (50, 45, 15, "High Success Rate", "large"),
        (50, 5, 15, "Low Success Rate", "large"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (total_count, success_count, num_draws, title, regime) in enumerate(test_configs):
        ax = axes[idx]
        
        # Create distribution and sample
        dist = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        samples = dist.sample((50000,))  # Large sample for good statistics
        
        # Determine sampling method used
        use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
        method = "HRUA" if use_hrua else "Simple"
        
        # Calculate support range
        lower, upper = dist.support
        support_range = torch.arange(lower.item(), upper.item() + 1)
        
        # Calculate theoretical PMF
        theoretical_probs = torch.exp(dist.log_prob(support_range))
        
        # Create empirical histogram
        samples_np = samples.numpy()
        bin_edges = np.arange(lower.item() - 0.5, upper.item() + 1.5, 1)
        counts, _ = np.histogram(samples_np, bins=bin_edges, density=True)
        
        # Plot comparison
        x_pos = support_range.numpy()
        width = 0.35
        
        ax.bar(x_pos - width/2, counts, width, alpha=0.7, label='Empirical', color='skyblue')
        ax.bar(x_pos + width/2, theoretical_probs.numpy(), width, alpha=0.7, label='Theoretical', color='orange')
        
        # Calculate statistics
        empirical_mean = samples.float().mean().item()
        theoretical_mean = dist.mean.item()
        
        ax.set_title(f'{title}\n{method} Method (N={total_count}, K={success_count}, n={num_draws})')
        ax.set_xlabel('Number of Successes')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Emp. μ: {empirical_mean:.3f}\nTheo. μ: {theoretical_mean:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"  {title}: Empirical mean = {empirical_mean:.4f}, Theoretical mean = {theoretical_mean:.4f}")
    
    plt.tight_layout()
    plt.savefig('hypergeometric_verification.png', dpi=150, bbox_inches='tight')
    print("  Plots saved to 'hypergeometric_verification.png'")
    
    # Test extreme large regime performance
    print("\n  Testing extreme large regime...")
    dist_extreme = Hypergeometric(total_count=10000, success_count=3000, num_draws=500)
    
    import time
    start_time = time.time()
    extreme_samples = dist_extreme.sample((10000,))
    extreme_time = time.time() - start_time
    
    extreme_mean = extreme_samples.float().mean().item()
    extreme_theoretical = dist_extreme.mean.item()
    
    print(f"  Extreme regime (N=10000, K=3000, n=500):")
    print(f"    Sampling time for 10k samples: {extreme_time:.4f}s")
    print(f"    Empirical mean: {extreme_mean:.2f}")
    print(f"    Theoretical mean: {extreme_theoretical:.2f}")
    print(f"    Difference: {abs(extreme_mean - extreme_theoretical):.4f}")
    
    # Test algorithm boundary conditions
    print("\n  Testing algorithm boundary conditions...")
    boundary_configs = [
        (20, 8, 9, "Just below HRUA threshold (n=9)"),
        (20, 8, 10, "At HRUA threshold (n=10)"), 
        (20, 8, 11, "Just above HRUA threshold (n=11)"),
        (20, 8, 19, "Near upper boundary (n=19)"),
        (20, 8, 20, "At upper boundary (n=20)"),
    ]
    
    for total_count, success_count, num_draws, description in boundary_configs:
        dist = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        samples = dist.sample((5000,))
        use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
        method = "HRUA" if use_hrua else "Simple"
        
        empirical_mean = samples.float().mean().item()
        theoretical_mean = dist.mean.item()
        
        print(f"    {description}: {method} method, emp_mean={empirical_mean:.3f}, theo_mean={theoretical_mean:.3f}")


def compare_with_numpy():
    """Compare our PyTorch implementation with NumPy's hypergeometric sampler."""
    print("\nComparing PyTorch vs NumPy Hypergeometric Implementations")
    print("=" * 60)
    
    # Test configurations covering both regimes
    test_configs = [
        # Small regime tests (uses simple sampling in our implementation)
        (20, 7, 5, "Small: Simple sampling regime"),
        (15, 6, 8, "Small: Simple sampling regime"),  
        (25, 10, 9, "Small: Just below HRUA threshold"),
        
        # Large regime tests (uses HRUA sampling in our implementation)  
        (50, 15, 20, "Large: HRUA sampling regime"),
        (100, 30, 25, "Large: HRUA sampling regime"),
        (200, 80, 50, "Large: HRUA sampling regime"),
        (1000, 300, 100, "Large: Very large HRUA regime"),
        
        # Edge cases
        (30, 25, 15, "Edge: High success rate"),
        (30, 5, 15, "Edge: Low success rate"),
        (50, 25, 49, "Edge: Near upper boundary"),
    ]
    
    # Statistical comparison
    print("\nStatistical Comparison:")
    print("-" * 60)
    print(f"{'Configuration':<30} {'PyTorch Mean':<12} {'NumPy Mean':<12} {'Diff':<8} {'Method':<8}")
    print("-" * 60)
    
    all_pytorch_samples = []
    all_numpy_samples = []
    
    for total_count, success_count, num_draws, description in test_configs:
        # PyTorch implementation
        dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        samples_torch = dist_torch.sample((20000,))
        
        # NumPy implementation  
        samples_numpy = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=20000)
        
        # Calculate statistics
        mean_torch = samples_torch.float().mean().item()
        mean_numpy = samples_numpy.mean()
        mean_diff = abs(mean_torch - mean_numpy)
        
        # Determine which method our implementation used
        use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
        method = "HRUA" if use_hrua else "Simple"
        
        print(f"{description:<30} {mean_torch:<12.4f} {mean_numpy:<12.4f} {mean_diff:<8.4f} {method:<8}")
        
        # Store samples for distribution comparison
        all_pytorch_samples.append((samples_torch.numpy(), description, total_count, success_count, num_draws))
        all_numpy_samples.append((samples_numpy, description, total_count, success_count, num_draws))
    
    # Perform statistical tests
    print(f"\nStatistical Tests:")
    print("-" * 40)
    
    for i, (config) in enumerate(test_configs):
        total_count, success_count, num_draws, description = config
        samples_torch = all_pytorch_samples[i][0]
        samples_numpy = all_numpy_samples[i][0]
        
        # Create bins for histogram comparison
        bins = np.arange(min(samples_torch.min(), samples_numpy.min()), max(samples_torch.max(), samples_numpy.max()) + 2) - 0.5
        
        # Get empirical distributions
        hist_torch, _ = np.histogram(samples_torch, bins=bins, density=True)
        hist_numpy, _ = np.histogram(samples_numpy, bins=bins, density=True)
        
        # Chi-square test between PyTorch and NumPy (not against theoretical)
        # Convert to counts for chi-square test
        observed_torch_counts, _ = np.histogram(samples_torch, bins=bins)
        observed_numpy_counts, _ = np.histogram(samples_numpy, bins=bins)
        
        # Kolmogorov-Smirnov test between PyTorch and NumPy
        ks_stat, ks_pvalue = stats.ks_2samp(samples_torch, samples_numpy)
        
        # Only perform chi-square test if we have enough samples in each bin
        if (observed_torch_counts >= 5).sum() > 5 and (observed_numpy_counts >= 5).sum() > 5:
            chi2_stat, chi2_pvalue = stats.chisquare(observed_torch_counts + 1, observed_numpy_counts + 1)  # Add 1 to avoid zeros
        else:
            chi2_stat, chi2_pvalue = np.nan, np.nan
        
        use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
        method = "HRUA" if use_hrua else "Simple"
        
        print(f"{description[:25]:<25} | {method:<6} | KS p-val: {ks_pvalue:.4f} | χ² p-val: {chi2_pvalue:.4f}")
    
    # Visual comparison plots
    print(f"\nGenerating visual comparisons...")
    
    # Select a subset for detailed visual comparison
    visual_configs = [
        (test_configs[1], "Small Regime"),  # Small regime
        (test_configs[4], "Large Regime"),  # Large regime  
        (test_configs[6], "Very Large Regime"),  # Very large regime
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (config, regime_name) in enumerate(visual_configs):
        total_count, success_count, num_draws, description = config
        ax = axes[idx]
        
        # Get samples
        dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        samples_torch = dist_torch.sample((50000,)).numpy()
        samples_numpy = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=50000)
        
        # Create histograms
        min_val = min(samples_torch.min(), samples_numpy.min())
        max_val = max(samples_torch.max(), samples_numpy.max())
        bins = np.arange(min_val, max_val + 2) - 0.5
        
        ax.hist(samples_torch, bins=bins, alpha=0.6, label='PyTorch', density=True, color='skyblue')
        ax.hist(samples_numpy, bins=bins, alpha=0.6, label='NumPy', density=True, color='orange')
        
        use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
        method = "HRUA" if use_hrua else "Simple"
        
        ax.set_title(f'{regime_name}\n{method} Method (N={total_count}, K={success_count}, n={num_draws})')
        ax.set_xlabel('Number of Successes')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_torch = samples_torch.mean()
        mean_numpy = samples_numpy.mean()
        stats_text = f'PyTorch μ: {mean_torch:.3f}\nNumPy μ: {mean_numpy:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('pytorch_vs_numpy_comparison.png', dpi=150, bbox_inches='tight')
    print("  Visual comparison saved to 'pytorch_vs_numpy_comparison.png'")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print("-" * 40)
    
    perf_configs = [
        (50, 15, 20, 10000, "Medium scale"),
        (200, 80, 50, 10000, "Large scale"), 
        (1000, 300, 100, 5000, "Very large scale"),
    ]
    
    import time
    
    # Test both CPU and GPU if available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n{device_name.upper()} Performance:")
        print("-" * 25)
        
        for total_count, success_count, num_draws, n_samples, scale in perf_configs:
            # PyTorch timing
            dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
            dist_torch = dist_torch.to(device)
            
            # Warm up GPU if needed
            if device_name == 'cuda':
                _ = dist_torch.sample((100,))
                torch.cuda.synchronize()
            
            start_time = time.time()
            samples = dist_torch.sample((n_samples,))
            if device_name == 'cuda':
                torch.cuda.synchronize()
            torch_time = time.time() - start_time
            
            # NumPy timing (always on CPU)
            if device_name == 'cpu':
                start_time = time.time()
                _ = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=n_samples)
                numpy_time = time.time() - start_time
                
                speedup = numpy_time / torch_time if torch_time > 0 else float('inf')
                
                use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
                method = "HRUA" if use_hrua else "Simple"
                
                print(f"{scale:<15} | {method:<6} | PyTorch: {torch_time:.4f}s | NumPy: {numpy_time:.4f}s | Speedup: {speedup:.2f}x")
            else:
                use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
                method = "HRUA" if use_hrua else "Simple"
                print(f"{scale:<15} | {method:<6} | PyTorch: {torch_time:.4f}s")
    
    # Test very large batch performance on GPU if available
    if torch.cuda.is_available():
        print(f"\nLarge Batch GPU Performance:")
        print("-" * 30)
        
        large_batch_configs = [
            (100, 30, 25, 100000, "100k samples"),
            (200, 80, 50, 50000, "50k samples"),
            (1000, 300, 100, 10000, "10k samples"),
        ]
        
        device = torch.device('cuda')
        for total_count, success_count, num_draws, n_samples, description in large_batch_configs:
            dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
            dist_torch = dist_torch.to(device)
            
            # Warm up
            _ = dist_torch.sample((1000,))
            torch.cuda.synchronize()
            
            start_time = time.time()
            samples = dist_torch.sample((n_samples,))
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            samples_per_sec = n_samples / gpu_time
            use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
            method = "HRUA" if use_hrua else "Simple"
            
            print(f"{description:<15} | {method:<6} | Time: {gpu_time:.4f}s | Rate: {samples_per_sec:,.0f} samples/s")


def compare_against_theoretical():
    """Compare both PyTorch and NumPy implementations against theoretical PMF using scipy."""
    print("\nComparing Against Theoretical PMF (using scipy)")
    print("=" * 55)
    
    # Test configurations across different regimes
    test_configs = [
        # Small regime
        (20, 7, 5, "Small regime"),
        (15, 6, 8, "Small regime"), 
        
        # Medium regime  
        (50, 15, 20, "Medium regime"),
        (100, 30, 25, "Medium regime"),
        
        # Large regime
        (200, 80, 50, "Large regime"),
        (1000, 300, 100, "Large regime"),
        
        # Edge cases
        (30, 25, 15, "High success rate"),
        (30, 5, 15, "Low success rate"),
    ]
    
    # Test both CPU and GPU if available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n{device_name.upper()} Results:")
        print(f"{'Configuration':<25} {'Method':<6} {'PyTorch χ²':<12} {'NumPy χ²':<12} {'PyTorch KS':<12} {'NumPy KS':<12}")
        print("-" * 85)
        
        for total_count, success_count, num_draws, description in test_configs:
            # Use scipy for theoretical distribution (unbiased reference)
            theoretical_dist = stats.hypergeom(M=total_count, n=success_count, N=num_draws)
            
            # Determine support
            lower = max(0, num_draws - (total_count - success_count))
            upper = min(num_draws, success_count)
            support_range = np.arange(lower, upper + 1)
            
            # Get theoretical PMF from scipy
            theoretical_probs = theoretical_dist.pmf(support_range)
            
            # Generate large samples for good statistics
            n_samples = 100000
            
            # PyTorch samples
            dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
            dist_torch = dist_torch.to(device)
            
            # Warm up GPU if needed
            if device_name == 'cuda':
                _ = dist_torch.sample((100,))
                torch.cuda.synchronize()
            
            samples_torch = dist_torch.sample((n_samples,))
            if device_name == 'cuda':
                torch.cuda.synchronize()
            samples_torch = samples_torch.cpu().numpy()  # Move to CPU for scipy comparison
            
            # NumPy samples (always on CPU)
            samples_numpy = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=n_samples)
            
            # Create bins for histogram comparison
            bins = np.arange(lower, upper + 2) - 0.5
            
            # Get observed counts (not densities)
            observed_torch_counts, _ = np.histogram(samples_torch, bins=bins)
            observed_numpy_counts, _ = np.histogram(samples_numpy, bins=bins)
            
            # Expected counts from theoretical distribution
            expected_counts = theoretical_probs * n_samples
            
            # Chi-square test against theoretical (only for bins with sufficient expected counts)
            chi2_torch, p_torch = stats.chisquare(observed_torch_counts, expected_counts)
            chi2_numpy, p_numpy = stats.chisquare(observed_numpy_counts, expected_counts)

            
            # KS test against theoretical CDF
            def theoretical_cdf(x):
                return theoretical_dist.cdf(x)
            
            # KS test for PyTorch
            ks_torch, ks_p_torch = stats.kstest(samples_torch, theoretical_cdf)
            
            # KS test for NumPy  
            ks_numpy, ks_p_numpy = stats.kstest(samples_numpy, theoretical_cdf)
            
            # Determine which method our implementation used
            use_hrua = (num_draws >= 10) and (num_draws <= total_count - 10)
            method = "HRUA" if use_hrua else "Simple"
            
            print(f"{description:<25} {method:<6} {p_torch:<12.4f} {p_numpy:<12.4f} {ks_p_torch:<12.4f} {ks_p_numpy:<12.4f}")
    
    # Detailed analysis for a few specific cases (test on primary device)
    primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDetailed Analysis (on {primary_device.type.upper()}):")
    print("-" * 50)
    
    detailed_configs = [
        (20, 7, 5, "Small: Simple sampling"),
        (100, 30, 25, "Large: HRUA sampling"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for config_idx, (total_count, success_count, num_draws, title) in enumerate(detailed_configs):
        # Use scipy for theoretical distribution
        theoretical_dist = stats.hypergeom(M=total_count, n=success_count, N=num_draws)
        
        # Determine support
        lower = max(0, num_draws - (total_count - success_count))
        upper = min(num_draws, success_count)
        support_range = np.arange(lower, upper + 1)
        theoretical_probs = theoretical_dist.pmf(support_range)
        
        # Generate samples
        n_samples = 50000
        
        # PyTorch samples
        dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        dist_torch = dist_torch.to(primary_device)
        
        if primary_device.type == 'cuda':
            _ = dist_torch.sample((100,))
            torch.cuda.synchronize()
        
        samples_torch = dist_torch.sample((n_samples,))
        if primary_device.type == 'cuda':
            torch.cuda.synchronize()
        samples_torch = samples_torch.cpu().numpy()
        
        # NumPy samples
        samples_numpy = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=n_samples)
        
        # Plot distributions
        ax1 = axes[config_idx, 0]
        ax2 = axes[config_idx, 1]
        
        # Histogram comparison
        bins = np.arange(lower, upper + 2) - 0.5
        x_pos = support_range
        
        ax1.hist(samples_torch, bins=bins, alpha=0.6, density=True, label='PyTorch', color='skyblue')
        ax1.hist(samples_numpy, bins=bins, alpha=0.6, density=True, label='NumPy', color='orange')
        ax1.bar(x_pos, theoretical_probs, alpha=0.8, width=0.1, label='Theoretical (scipy)', color='red')
        
        ax1.set_title(f'{title} - Distribution Comparison')
        ax1.set_xlabel('Number of Successes')
        ax1.set_ylabel('Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error analysis - absolute difference from theoretical
        hist_torch, _ = np.histogram(samples_torch, bins=bins, density=True)
        hist_numpy, _ = np.histogram(samples_numpy, bins=bins, density=True)
        
        error_torch = np.abs(hist_torch - theoretical_probs)
        error_numpy = np.abs(hist_numpy - theoretical_probs)
        
        ax2.bar(x_pos - 0.2, error_torch, width=0.4, alpha=0.7, label='PyTorch Error', color='skyblue')
        ax2.bar(x_pos + 0.2, error_numpy, width=0.4, alpha=0.7, label='NumPy Error', color='orange')
        
        ax2.set_title(f'{title} - Absolute Error from Theoretical')
        ax2.set_xlabel('Number of Successes')
        ax2.set_ylabel('|Empirical - Theoretical|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calculate and display summary statistics
        mse_torch = np.mean(error_torch**2)
        mse_numpy = np.mean(error_numpy**2)
        mae_torch = np.mean(error_torch)
        mae_numpy = np.mean(error_numpy)
        
        print(f"\n{title}:")
        print(f"  PyTorch MSE: {mse_torch:.6f}, MAE: {mae_torch:.6f}")
        print(f"  NumPy MSE:   {mse_numpy:.6f}, MAE: {mae_numpy:.6f}")
        print(f"  PyTorch {'BETTER' if mse_torch < mse_numpy else 'WORSE'} than NumPy (MSE)")
    
    plt.tight_layout()
    plt.savefig('theoretical_comparison_scipy.png', dpi=150, bbox_inches='tight')
    print(f"\nDetailed plots saved to 'theoretical_comparison_scipy.png'")
    
    # Summary statistics across all configurations  
    print(f"\nSummary Analysis (using scipy reference):")
    print("-" * 40)
    
    torch_better_chi2 = 0
    torch_better_ks = 0
    total_valid_tests = 0
    
    # Use primary device for summary
    for total_count, success_count, num_draws, description in test_configs:
        theoretical_dist = stats.hypergeom(M=total_count, n=success_count, N=num_draws)
        
        lower = max(0, num_draws - (total_count - success_count))
        upper = min(num_draws, success_count)
        support_range = np.arange(lower, upper + 1)
        theoretical_probs = theoretical_dist.pmf(support_range)
        
        n_samples = 50000
        
        # PyTorch samples
        dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        dist_torch = dist_torch.to(primary_device)
        samples_torch = dist_torch.sample((n_samples,)).cpu().numpy()
        
        # NumPy samples
        samples_numpy = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=n_samples)
        
        # Calculate errors
        bins = np.arange(lower, upper + 2) - 0.5
        hist_torch, _ = np.histogram(samples_torch, bins=bins, density=True)
        hist_numpy, _ = np.histogram(samples_numpy, bins=bins, density=True)
        
        error_torch = np.abs(hist_torch - theoretical_probs)
        error_numpy = np.abs(hist_numpy - theoretical_probs)
        
        mse_torch = np.mean(error_torch**2)
        mse_numpy = np.mean(error_numpy**2)
        
        if mse_torch < mse_numpy:
            torch_better_chi2 += 1
            
        # KS test comparison
        ks_torch, _ = stats.kstest(samples_torch, theoretical_dist.cdf)
        ks_numpy, _ = stats.kstest(samples_numpy, theoretical_dist.cdf)
        
        if ks_torch < ks_numpy:  # Lower KS statistic is better
            torch_better_ks += 1
            
        total_valid_tests += 1
    
    print(f"PyTorch better than NumPy in {torch_better_chi2}/{total_valid_tests} cases (MSE)")
    print(f"PyTorch better than NumPy in {torch_better_ks}/{total_valid_tests} cases (KS statistic)")
    
    if torch.cuda.is_available():
        print(f"Note: GPU performance tested and results saved for comparison")


def test_optimized_performance():
    """Test performance improvements from compilation and vectorization."""
    print("\nTesting Optimized Performance (Simplified)")
    print("=" * 50)
    
    # Test configurations covering both sampling regimes
    test_configs = [
        # Small regime (simple sampling) - now fully vectorized
        (20, 7, 3, 10000, "Tiny: Fast simple"),
        (25, 10, 4, 10000, "Small: Fast simple"),
        (30, 12, 4, 10000, "Medium small: Fast simple"),
        
        # Medium regime (HRUA with lower threshold)
        (50, 15, 8, 10000, "Medium: Fast HRUA"),
        (100, 30, 15, 10000, "Large: Fast HRUA"),
        (200, 80, 30, 5000, "Very large: Fast HRUA"),
        (1000, 300, 100, 2000, "Massive: Fast HRUA"),
    ]
    
    # Test both CPU and GPU if available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    import time
    
    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n{device_name.upper()} Performance:")
        print("-" * 45)
        print(f"{'Configuration':<25} {'Method':<8} {'Time (s)':<10} {'Rate (k/s)':<12} {'vs NumPy':<10}")
        print("-" * 80)
        
        for total_count, success_count, num_draws, n_samples, description in test_configs:
            # Determine which method will be used
            use_hrua = (num_draws >= 5) and (num_draws <= total_count - 5)
            method = "HRUA" if use_hrua else "Simple"
            
            # PyTorch timing
            dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
            dist_torch = dist_torch.to(device)
            
            # Proper warmup for compilation
            if device_name == 'cuda':
                for _ in range(5):
                    _ = dist_torch.sample((100,))
                torch.cuda.synchronize()
            else:
                # CPU warmup for compilation
                for _ in range(5):
                    _ = dist_torch.sample((100,))
            
            start_time = time.time()
            samples = dist_torch.sample((n_samples,))
            if device_name == 'cuda':
                torch.cuda.synchronize()
            torch_time = time.time() - start_time
            
            torch_rate = n_samples / torch_time / 1000  # samples per second in thousands
            
            # NumPy timing (only on CPU)
            if device_name == 'cpu':
                start_time = time.time()
                _ = np.random.hypergeometric(success_count, total_count - success_count, num_draws, size=n_samples)
                numpy_time = time.time() - start_time
                
                speedup = numpy_time / torch_time if torch_time > 0 else float('inf')
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{description:<25} {method:<8} {torch_time:<10.4f} {torch_rate:<12.1f} {speedup_str:<10}")
    
    # Compilation benefit analysis
    print(f"\nCompilation Benefits Analysis:")
    print("-" * 40)
    
    primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    comparison_configs = [
        (20, 7, 4, 5000, "Small regime"),
        (50, 15, 10, 5000, "Medium regime"),  
        (200, 80, 30, 2000, "Large regime"),
    ]
    
    print(f"Testing on {primary_device.type.upper()}:")
    print(f"{'Configuration':<15} {'No Compile (s)':<15} {'Compiled (s)':<15} {'Speedup':<10} {'Method':<8}")
    print("-" * 75)
    
    for total_count, success_count, num_draws, n_samples, description in comparison_configs:
        # Test without compilation (disable torch.compile)
        global USE_COMPILE
        old_use_compile = USE_COMPILE
        USE_COMPILE = False
        
        dist_no_compile = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        dist_no_compile = dist_no_compile.to(primary_device)
        
        # Warmup
        for _ in range(3):
            _ = dist_no_compile.sample((100,))
        if primary_device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = dist_no_compile.sample((n_samples,))
        if primary_device.type == 'cuda':
            torch.cuda.synchronize()
        no_compile_time = time.time() - start_time
        
        # Test with compilation
        USE_COMPILE = old_use_compile
        
        dist_compiled = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
        dist_compiled = dist_compiled.to(primary_device)
        
        # Warmup with compilation
        for _ in range(5):
            _ = dist_compiled.sample((100,))
        if primary_device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = dist_compiled.sample((n_samples,))
        if primary_device.type == 'cuda':
            torch.cuda.synchronize()
        compiled_time = time.time() - start_time
        
        speedup = no_compile_time / compiled_time if compiled_time > 0 else float('inf')
        
        # Determine method used
        use_hrua = (num_draws >= 5) and (num_draws <= total_count - 5)
        method = "HRUA" if use_hrua else "Simple"
        
        print(f"{description:<15} {no_compile_time:<15.4f} {compiled_time:<15.4f} {speedup:<10.2f} {method:<8}")
        
        # Restore original setting
        USE_COMPILE = old_use_compile


def test_log_pmf_validation():
    """Comprehensive validation of log PMF against scipy.stats.hypergeom."""
    print("\nComprehensive Log PMF Validation Against Scipy")
    print("=" * 55)
    
    # Comprehensive test configurations covering edge cases and normal cases
    test_configs = [
        # Small values
        (5, 2, 3, "Tiny case"),
        (10, 4, 5, "Small case"),
        (20, 7, 12, "Medium case"),
        
        # Edge cases - boundary conditions
        (10, 0, 5, "No successes available"),
        (10, 10, 5, "All successes"),
        (10, 5, 0, "No draws"),
        (10, 5, 10, "Draw everything"),
        (10, 3, 8, "More draws than failures"),
        
        # Large values
        (100, 30, 25, "Large case"),
        (1000, 300, 100, "Very large case"),
        (500, 450, 200, "High success rate"),
        (500, 50, 200, "Low success rate"),
        
        # Extreme cases
        (1000, 1, 500, "Very few successes"),
        (1000, 999, 500, "Almost all successes"),
        (2, 1, 1, "Minimal case"),
    ]
    
    print(f"{'Configuration':<25} {'Max Abs Diff':<15} {'Max Rel Diff':<15} {'Status':<10}")
    print("-" * 75)
    
    all_max_abs_diffs = []
    all_max_rel_diffs = []
    problem_cases = []
    
    for total_count, success_count, num_draws, description in test_configs:
        try:
            # Create our distribution
            dist_torch = Hypergeometric(total_count=total_count, success_count=success_count, num_draws=num_draws)
            
            # Create scipy distribution
            dist_scipy = stats.hypergeom(M=total_count, n=success_count, N=num_draws)
            
            # Determine support
            lower = max(0, num_draws - (total_count - success_count))
            upper = min(num_draws, success_count)
            
            if lower <= upper:
                support_values = torch.arange(lower, upper + 1)
                
                # Get log PMF from both
                torch_log_pmf = dist_torch.log_prob(support_values)
                scipy_log_pmf = dist_scipy.logpmf(support_values.numpy())
                
                # Convert to numpy for comparison
                torch_log_pmf_np = torch_log_pmf.numpy()
                
                # Calculate differences
                abs_diffs = np.abs(torch_log_pmf_np - scipy_log_pmf)
                
                # Handle -inf values carefully for relative differences
                finite_mask = np.isfinite(torch_log_pmf_np) & np.isfinite(scipy_log_pmf)
                if finite_mask.any():
                    rel_diffs = np.abs((torch_log_pmf_np[finite_mask] - scipy_log_pmf[finite_mask]) / 
                                     (scipy_log_pmf[finite_mask] + 1e-15))  # Add small epsilon to avoid division by zero
                    max_rel_diff = np.max(rel_diffs) if len(rel_diffs) > 0 else 0.0
                else:
                    max_rel_diff = 0.0
                
                max_abs_diff = np.max(abs_diffs)
                
                # Check if -inf values match
                inf_mismatch = False
                torch_inf = np.isinf(torch_log_pmf_np) & (torch_log_pmf_np < 0)
                scipy_inf = np.isinf(scipy_log_pmf) & (scipy_log_pmf < 0)
                if not np.array_equal(torch_inf, scipy_inf):
                    inf_mismatch = True
                
                # Determine status
                if inf_mismatch:
                    status = "INF MISMATCH"
                    problem_cases.append((description, total_count, success_count, num_draws, "inf_mismatch"))
                elif max_abs_diff > 1e-10:
                    status = "HIGH DIFF"
                    problem_cases.append((description, total_count, success_count, num_draws, f"abs_diff={max_abs_diff:.2e}"))
                elif max_rel_diff > 1e-10:
                    status = "HIGH REL"
                    problem_cases.append((description, total_count, success_count, num_draws, f"rel_diff={max_rel_diff:.2e}"))
                else:
                    status = "GOOD"
                
                all_max_abs_diffs.append(max_abs_diff)
                all_max_rel_diffs.append(max_rel_diff)
                
                print(f"{description:<25} {max_abs_diff:<15.2e} {max_rel_diff:<15.2e} {status:<10}")
                
            else:
                print(f"{description:<25} {'INVALID':<15} {'INVALID':<15} {'BAD RANGE':<10}")
                problem_cases.append((description, total_count, success_count, num_draws, "invalid_range"))
                
        except Exception as e:
            print(f"{description:<25} {'ERROR':<15} {'ERROR':<15} {'EXCEPTION':<10}")
            problem_cases.append((description, total_count, success_count, num_draws, f"exception: {e}"))
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print("-" * 30)
    if all_max_abs_diffs:
        print(f"Max absolute difference across all cases: {max(all_max_abs_diffs):.2e}")
        print(f"Mean absolute difference: {np.mean(all_max_abs_diffs):.2e}")
        print(f"Max relative difference across all cases: {max(all_max_rel_diffs):.2e}")
        print(f"Mean relative difference: {np.mean(all_max_rel_diffs):.2e}")
    
    print(f"Total test cases: {len(test_configs)}")
    print(f"Problem cases: {len(problem_cases)}")
    
    # Detailed analysis of problem cases
    if problem_cases:
        print(f"\nDetailed Problem Case Analysis:")
        print("-" * 40)
        
        for desc, N, K, n, issue in problem_cases:
            print(f"\nProblem: {desc} (N={N}, K={K}, n={n})")
            print(f"Issue: {issue}")
            
            try:
                # Show detailed comparison for this case
                dist_torch = Hypergeometric(total_count=N, success_count=K, num_draws=n)
                dist_scipy = stats.hypergeom(M=N, n=K, N=n)
                
                lower = max(0, n - (N - K))
                upper = min(n, K)
                
                if lower <= upper:
                    support_values = torch.arange(lower, upper + 1)
                    torch_log_pmf = dist_torch.log_prob(support_values)
                    scipy_log_pmf = dist_scipy.logpmf(support_values.numpy())
                    
                    print(f"Support: {support_values.tolist()}")
                    print(f"PyTorch log PMF: {torch_log_pmf}")
                    print(f"Scipy log PMF:   {scipy_log_pmf}")
                    print(f"Differences:     {torch_log_pmf - scipy_log_pmf}")
                    
                    # Also show regular PMF for easier interpretation
                    torch_pmf = np.exp(torch_log_pmf.numpy())
                    scipy_pmf = np.exp(scipy_log_pmf)
                    print(f"PyTorch PMF: {torch_pmf}")
                    print(f"Scipy PMF:   {scipy_pmf}")
                    print(f"PMF sum (PyTorch): {torch_pmf.sum():.10f}")
                    print(f"PMF sum (Scipy):   {scipy_pmf.sum():.10f}")
            except Exception as e:
                print(f"Error in detailed analysis: {e}")
    
    # Test specific edge cases that might cause issues
    print(f"\nSpecial Edge Case Tests:")
    print("-" * 30)
    
    edge_cases = [
        # Cases where support might be empty or single-valued
        (1, 0, 1, "Support: [0]"),
        (1, 1, 1, "Support: [1]"), 
        (2, 1, 2, "Support: [0,1]"),
        (3, 0, 2, "Support: [0]"),
        (3, 3, 2, "Support: [2]"),
        (100, 0, 50, "Large N, K=0"),
        (100, 100, 50, "Large N, K=N"),
    ]
    
    for N, K, n, desc in edge_cases:
        try:
            dist_torch = Hypergeometric(total_count=N, success_count=K, num_draws=n)
            dist_scipy = stats.hypergeom(M=N, n=K, N=n)
            
            lower = max(0, n - (N - K))
            upper = min(n, K)
            support_values = torch.arange(lower, upper + 1)
            
            torch_log_pmf = dist_torch.log_prob(support_values)
            scipy_log_pmf = dist_scipy.logpmf(support_values.numpy())
            
            max_diff = np.max(np.abs(torch_log_pmf - scipy_log_pmf))
            
            status = "GOOD" if max_diff < 1e-12 else f"DIFF={max_diff:.2e}"
            print(f"{desc:<20} (N={N}, K={K}, n={n}): {status}")
            
        except Exception as e:
            print(f"{desc:<20} (N={N}, K={K}, n={n}): ERROR - {e}")


if __name__ == "__main__":
    test_hypergeometric()
    compare_with_numpy()
    compare_against_theoretical()
    test_optimized_performance()
    test_log_pmf_validation() 