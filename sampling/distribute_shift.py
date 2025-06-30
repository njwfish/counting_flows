import torch

def get_proportional_weighted_dist(x0_hat_t):
    """
    Get a weighted distribution for sampling proportional to the values in x0_hat_t.
    """
    total_sum = x0_hat_t.sum()
    if total_sum == 0:
        return torch.ones_like(x0_hat_t) / x0_hat_t.numel()
    
    return x0_hat_t / total_sum

def sample_pert(x0_hat_t, weighted_dist, mu_diff):
    """
    Sample a perturbation and apply it to x0_hat_t to match the target mean difference.
    """
    B, d = x0_hat_t.shape
    total_diff = (mu_diff * B).sum().round().long()

    if total_diff == 0:
        return x0_hat_t

    pert = torch.zeros_like(x0_hat_t)
    
    if total_diff > 0:
        # Sample indices to increment
        flat_dist = weighted_dist.flatten()
        if flat_dist.sum() > 0:
            indices = torch.multinomial(flat_dist, total_diff, replacement=True)
            pert.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=pert.dtype))
        else: # uniform sampling if weights are all zero
            indices = torch.randint(0, B * d, (total_diff,), device=x0_hat_t.device)
            pert.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=pert.dtype))
            
    elif total_diff < 0:
        # Sample indices to decrement
        flat_dist = weighted_dist.flatten()
        if flat_dist.sum() > 0:
            indices = torch.multinomial(flat_dist, -total_diff, replacement=True)
            pert.view(-1).scatter_add_(0, indices, -torch.ones_like(indices, dtype=pert.dtype))
        else: # uniform sampling if weights are all zero
            indices = torch.randint(0, B * d, (-total_diff,), device=x0_hat_t.device)
            pert.view(-1).scatter_add_(0, indices, -torch.ones_like(indices, dtype=pert.dtype))

    x0_hat_t_pert = x0_hat_t + pert
    # Ensure non-negativity
    x0_hat_t_pert = torch.clamp(x0_hat_t_pert, min=0)
    
    # Adjust to match the exact total_diff
    current_total = x0_hat_t_pert.sum()
    target_total = x0_hat_t.sum() + total_diff
    adjustment_diff = target_total - current_total
    
    while adjustment_diff != 0:
        if adjustment_diff > 0:
            # Need to add more counts
            idx = torch.randint(0, B * d, (adjustment_diff.abs().long(),), device=x0_hat_t.device)
            x0_hat_t_pert.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=pert.dtype))
        else:
            # Need to remove counts
            non_zero_indices = (x0_hat_t_pert.view(-1) > 0).nonzero().squeeze()
            if len(non_zero_indices) > 0:
                idx_to_sample = non_zero_indices
                num_to_sample = min(adjustment_diff.abs().long(), len(idx_to_sample))
                if num_to_sample > 0:
                   idx = idx_to_sample[torch.randint(0, len(idx_to_sample), (num_to_sample,))]
                   x0_hat_t_pert.view(-1).scatter_add_(0, idx, -torch.ones_like(idx, dtype=pert.dtype))
            
        current_total = x0_hat_t_pert.sum()
        new_adjustment_diff = target_total - current_total
        if new_adjustment_diff == adjustment_diff: # break if no progress is made
            break
        adjustment_diff = new_adjustment_diff

    return x0_hat_t_pert 