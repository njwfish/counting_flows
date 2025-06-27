import numpy as np

def get_proportional_weighted_dist(X):
    dist_sum = X.sum(axis=0)
    weighted_dist = (X / np.clip(dist_sum, 1, None))
    weighted_dist[:, dist_sum == 0] = 1
    weighted_dist = weighted_dist.astype(np.float64)
    weighted_dist /= weighted_dist.sum(axis=0)
    return weighted_dist


def sample_multinomial_batch(probs, counts, max_count=None, max_rejections=100):
    """
    Vectorized multinomial sampling for batches with different total counts
    
    Args:
        probs: tensor of shape (batch_size, num_categories)
        counts: tensor of shape (num_categories,) containing total counts
    """
    result = np.zeros_like(probs)
    for i, (p, c) in enumerate(zip(probs.T, counts)):
        p = p.copy()
        if c == 0 or p.sum() == 0:
            continue
        elif max_count is not None and c > max_count[:, i].sum():
            result[:, i] = max_count[:, i]
        
        s = np.random.multinomial(int(c), p)

        if max_count is not None:
            mc = max_count[:, i]
            over_max = s > mc
            num_rejections = 0
            while np.any(over_max):
                p[over_max] = 0
                p = p / p.sum()
                n_resample = np.sum(s[over_max] - mc[over_max])
                s[over_max] = mc[over_max]
                # assert np.sum(s) + n_resample == c
                s += np.random.multinomial(n_resample.astype(np.int64), p)

                over_max = s > mc
                num_rejections += 1
                if num_rejections > max_rejections:
                    break
        result[:, i] = s
    
    return result

def sample_pert(ctrl, weighted_dist, mean_shift, max_rejections=100):
    count_shift = np.round(mean_shift * ctrl.shape[0])
    max_count = 1. * ctrl
    max_count[:, count_shift > 0] = np.inf
    samples = sample_multinomial_batch(
        weighted_dist, np.abs(count_shift), max_count=max_count, max_rejections=max_rejections
    )
    sampled_pert = ctrl + np.sign(count_shift) * samples
    sampled_pert = np.clip(sampled_pert, 0, None)
    return sampled_pert

def hard_sample_pert(ctrl, weighted_dist, mean_shift, round_pert=False):
    # this can probably be better too. really we should like solve a linear program to find the closest integer solution
    count_shift = np.round(mean_shift * ctrl.shape[0])
    sampled_pert = ctrl + np.round(count_shift * weighted_dist) if round_pert else ctrl + (count_shift * weighted_dist)
    sampled_pert = np.clip(sampled_pert, 0, None)
    return sampled_pert