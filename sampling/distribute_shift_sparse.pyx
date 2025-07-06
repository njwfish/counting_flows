# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix
from numpy.random import default_rng
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t
ctypedef np.intp_t   INTP_t

def adjust_sparse_counts_cython_colwise(object X_csr,
                                        np.ndarray[DTYPE_t, ndim=1] S):
    """
    Column-wise version: adjusts sparse matrix to hit target column sums.
    Uses CSC format internally for efficient column access.
    
    Args:
        X_csr: CSR sparse matrix
        S: target column sums (length = n_cols)
    
    Returns:
        X_updated: CSR matrix with adjusted column sums
    """
    # Convert to CSC format for efficient column operations
    from scipy.sparse import csc_matrix
    X_csc = X_csr.tocsc()
    
    # Create properly typed arrays for cdef
    cdef np.ndarray[INTP_t, ndim=1] indptr_array = X_csc.indptr.astype(np.intp)
    cdef np.ndarray[INTP_t, ndim=1] indices_array = X_csc.indices.astype(np.intp)
    cdef np.ndarray[DTYPE_t, ndim=1] data_array = X_csc.data.astype(DTYPE)
    
    cdef:
        INTP_t[:] indptr  = indptr_array  # Column pointers
        INTP_t[:] indices = indices_array # Row indices
        DTYPE_t[:] data   = data_array
        DTYPE_t[:] Sview  = S
        Py_ssize_t n      = X_csc.shape[0]  # n_rows
        Py_ssize_t d      = X_csc.shape[1]  # n_cols
        Py_ssize_t col, start, end, j
        long      D_col
        double    total_col
        int       sign
        
    # Initialize RNG
    rng = default_rng()
    
    # First pass: compute current column sums (efficient with CSC)
    cdef np.ndarray[DTYPE_t, ndim=1] current_col_sums = np.zeros(d, dtype=DTYPE)
    for col in range(d):
        start = indptr[col]
        end = indptr[col + 1]
        for j in range(start, end):
            current_col_sums[col] += data[j]
    
    # Copy original data (we'll modify it in place)
    cdef np.ndarray[DTYPE_t, ndim=1] new_data = np.array(data, dtype=DTYPE)
    
    # Process each column that needs adjustment
    for col in range(d):
        D_col = <long>current_col_sums[col] - <long>Sview[col]
        
        if D_col != 0:
            m = abs(D_col)
            sign = -1 if D_col > 0 else 1
            
            # Get column range in CSC format
            start = indptr[col]
            end = indptr[col + 1]
            col_len = end - start
            
            if col_len > 0:
                # Extract column values efficiently
                col_values = []
                for j in range(start, end):
                    col_values.append(data[j])
                
                total_col = sum(col_values)
                
                if total_col > 0:
                    # Use current column values as probabilities
                    probs = np.array(col_values, dtype=np.float64) / total_col
                    
                    # Sample adjustments
                    adjustments = rng.multinomial(m, probs)
                    
                    # Apply adjustments directly to CSC data
                    for j in range(col_len):
                        if adjustments[j] > 0:
                            new_data[start + j] += sign * adjustments[j]
    
    # Create updated CSC matrix and convert back to CSR
    X_csc_updated = csc_matrix((new_data, indices, indptr), shape=(n, d), dtype=DTYPE)
    X_csc_updated.eliminate_zeros()
    
    # Convert back to CSR format
    X_updated = X_csc_updated.tocsr()
    return X_updated

# Keep the original row-wise function for backwards compatibility
def adjust_sparse_counts_cython(object X_csr,
                                np.ndarray[DTYPE_t, ndim=1] S):
    """
    Original row-wise version for backwards compatibility
    """
    cdef:
        INTP_t[:] indptr  = X_csr.indptr
        INTP_t[:] indices = X_csr.indices
        DTYPE_t[:] data   = X_csr.data
        DTYPE_t[:] Sview  = S
        Py_ssize_t n      = X_csr.shape[0]
        Py_ssize_t d      = X_csr.shape[1]
        Py_ssize_t i, start, end, p_len, j
        long      D_i
        double    total_local
        int       sign
        np.ndarray[np.float64_t, ndim=1] p
    # initialize a single RNG
    rng = default_rng()
    data_out    = []
    indices_out = []
    indptr_out  = [0]
    for i in range(n):
        start = indptr[i]
        end   = indptr[i+1]
        # compute row sum
        total_local = 0.0
        for j in range(start, end):
            total_local += data[j]
        D_i = <long>total_local - <long>Sview[i]
        if D_i != 0:
            m = abs(D_i)
            sign = -1 if D_i > 0 else 1
            p_len = end - start
            if p_len > 0 and total_local > 0.0:
                # 1) build a small NumPy view of the row's weights
                p = np.empty(p_len, dtype=np.float64)
                for j in range(p_len):
                    p[j] = data[start + j]
                p /= p.sum()
                # 2) call NumPy's multinomial sampler!
                #    returns a 1‑D array of length p_len
                s = rng.multinomial(m, p)
                # 3) record only non‑zero draws
                for j in range(p_len):
                    cnt = s[j]
                    if cnt:
                        data_out.append(sign * int(cnt))
                        indices_out.append(indices[start + j])
            else:
                # fallback to uniform over all d columns
                s = rng.multinomial(m, np.ones(d)/d)
                for j in range(d):
                    cnt = s[j]
                    if cnt:
                        data_out.append(sign * int(cnt))
                        indices_out.append(j)
        indptr_out.append(len(data_out))
    # assemble delta and apply
    delta = csr_matrix(
        (data_out, indices_out, indptr_out),
        shape=(n, d), dtype=DTYPE
    )
    X_updated = X_csr + delta
    X_updated.eliminate_zeros()
    return X_updated