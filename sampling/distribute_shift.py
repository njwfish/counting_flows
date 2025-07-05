import torch

def get_proportional_weighted_dist(x0_hat_t):
    """
    Get a weighted distribution for sampling proportional to the values in x0_hat_t.
    Column-wise normalization to match numpy implementation.
    """
    # Sum along axis 0 (rows) to get column sums - shape (d,)
    dist_sum = x0_hat_t.sum(axis=0)
    
    # Avoid division by zero by clipping to minimum value of 1
    dist_sum_clipped = torch.clamp(dist_sum, min=1.0)
    
    # Divide each column by its sum
    weighted_dist = x0_hat_t / dist_sum_clipped
    
    # Set columns with zero sum to uniform distribution (1/n_rows)
    zero_sum_cols = (dist_sum == 0)
    if zero_sum_cols.any():
        n_rows = x0_hat_t.shape[0]
        weighted_dist[:, zero_sum_cols] = 1.0 / n_rows
    
    # Convert to float64 for precision and renormalize column-wise
    weighted_dist = weighted_dist.double()
    weighted_dist = weighted_dist / weighted_dist.sum(axis=0)
    
    return weighted_dist

def sample_pert(x0_hat_t, weighted_dist, mu_diff):
    """
    Sample a perturbation and apply it to x0_hat_t to match the target mean difference.
    Fully vectorized implementation using scatter_add and multinomial sampling.
    """
    B, d = x0_hat_t.shape
    device = x0_hat_t.device
    
    # Calculate count shift per column (like numpy version)
    count_shift = torch.round(mu_diff * B)
    
    if torch.all(count_shift == 0):
        return x0_hat_t
    
    # Split into positive and negative shifts
    pos_shift = torch.clamp(count_shift, min=0).long()
    neg_shift = torch.clamp(-count_shift, min=0).long()
    
    samples = torch.zeros_like(x0_hat_t)
    
    # Handle positive shifts (adding counts) - fully vectorized
    total_pos = pos_shift.sum()
    if total_pos > 0:
        # Create weighted flat probabilities
        # Each column contributes pos_shift[j] copies of its probabilities
        flat_probs = weighted_dist * pos_shift.float()
        flat_probs = flat_probs.flatten()
        
        if flat_probs.sum() > 0:
            # Sample all at once
            indices = torch.multinomial(flat_probs, total_pos, replacement=True)
            samples.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=samples.dtype))
    
    # Handle negative shifts (removing counts) - with constraints
    neg_cols = torch.where(neg_shift > 0)[0]
    for j in neg_cols:
        c = neg_shift[j]
        probs = weighted_dist[:, j]
        max_count = x0_hat_t[:, j]
        
        if probs.sum() > 0:
            # Rejection sampling for max count constraints
            temp_samples = torch.zeros_like(max_count)
            remaining = c
            max_rejections = 100
            
            for _ in range(max_rejections):
                if remaining <= 0:
                    break
                    
                # Sample indices
                indices = torch.multinomial(probs, remaining, replacement=True)
                temp_samples.scatter_add_(0, indices, torch.ones_like(indices, dtype=temp_samples.dtype))
                
                # Check max count violations
                over_max = temp_samples > max_count
                if over_max.any():
                    # Calculate excess and adjust
                    excess = (temp_samples[over_max] - max_count[over_max]).sum()
                    temp_samples[over_max] = max_count[over_max]
                    remaining = excess.long()
                    
                    # Update probabilities to avoid over-max elements
                    probs = probs.clone()
                    probs[over_max] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    else:
                        break
                else:
                    break
            
            samples[:, j] = temp_samples
    
    # Apply the perturbation with correct signs
    sampled_pert = x0_hat_t + torch.sign(count_shift) * samples
    
    # Ensure non-negativity
    sampled_pert = torch.clamp(sampled_pert, min=0)
    
    return sampled_pert 