import numpy as np
from scipy.spatial.distance import jensenshannon

def jsd_ct(predicted: np.ndarray, true: np.ndarray, eps: float = 1e-12):
    """
    Paper Eq. (2): JSD per cell type k between Q(P_k) and Q(T_k),
    where each column is normalized across spots.

    Parameters
    ----------
    predicted, true : arrays of shape (n_spots, n_celltypes)

    Returns
    -------
    jsd_per_ct : (n_celltypes,) JSD for each cell type (in nats)
    jsd_mean   : scalar mean across cell types
    """
    if predicted.shape != true.shape:
        raise ValueError("Arrays must have the same shape.")

    # stabilize and normalize each column to sum to 1 (distribution over spots)
    P = predicted + eps
    P /= P.sum(axis=0, keepdims=True)
    Q = true + eps
    Q /= Q.sum(axis=0, keepdims=True)

    M = 0.5 * (P + Q)

    # KL terms summed over spots (axis=0), yielding one value per cell type
    jsd_per_ct = 0.5 * np.sum(P * (np.log(P) - np.log(M)), axis=0) + \
                 0.5 * np.sum(Q * (np.log(Q) - np.log(M)), axis=0)

    return jsd_per_ct, float(jsd_per_ct.mean())

def mean_jsd(predicted: np.ndarray, true: np.ndarray):
    """
    Compute Jensen–Shannon divergence row-wise between two arrays 
    of same shape and return all values and their mean.

    Parameters
    ----------
    predicted : np.ndarray
        2D array (n_rows x n_features).
    true : np.ndarray
        2D array of same shape as predicted.

    Returns
    -------
    jsd_values : np.ndarray
        Array of shape (n_rows,) with JSD values for each row.
    jsd_mean : float
        Mean JSD across all rows.
    """
    if predicted.shape != true.shape:
        raise ValueError("Arrays must have the same shape.")

    jsd_values = []
    for i in range(predicted.shape[0]):
        # normalize rows to sum to 1 (to be valid probability distributions)
        p = predicted[i] / np.sum(predicted[i])
        q = true[i] / np.sum(true[i])
        jsd = jensenshannon(p, q, base=2) ** 2  # square because scipy returns sqrt(JSD)
        jsd_values.append(jsd)

    jsd_values = np.array(jsd_values)
    return jsd_values, jsd_values.mean()


import numpy as np

def rmse(predicted: np.ndarray, true: np.ndarray):
    """
    Compute squared error per column, normalize by column sum,
    then average across columns.

    Parameters
    ----------
    predicted : np.ndarray
        2D array (n_rows x n_cols).
    true : np.ndarray
        2D array of same shape.

    Returns
    -------
    float
        Average normalized squared error across columns.
    """
    if predicted.shape != true.shape:
        raise ValueError("Arrays must have the same shape.")
    sq_err = (predicted - true) ** 2
    col_sums = true.sum(axis=0)
    col_scores = sq_err.sum(axis=0) / col_sums
    return np.sqrt(col_scores.mean())
