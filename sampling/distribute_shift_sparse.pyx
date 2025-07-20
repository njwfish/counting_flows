# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

# Python imports
import numpy as np
from numpy.random import default_rng
from scipy.sparse import csr_matrix, csc_matrix

# Cython imports
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Dtype aliases
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t
ctypedef np.intp_t   INTP_t
# 32-bit index types
ctypedef np.int32_t IDX_t
ctypedef np.int32_t PTR_t
ctypedef np.float32_t VAL_t
# single-token alias for unsigned int
ctypedef unsigned int uint

# Initialize the NumPy C API
np.import_array()

cdef extern from "multinom.hpp" nogil:
    void sample_multinomial_c(unsigned int* counts,
                              int K,
                              unsigned int n,
                              const double* probs) noexcept

# Python wrapper for one-off multinomial draws
def sample_multinomial(unsigned int n,
                       np.ndarray[np.float64_t, ndim=1] p):
    """
    Draw one Multinomial(n, p) sample outside the GIL.

    Args:
      n : total number of trials (non-negative integer)
      p : 1D NumPy array of probabilities (length K)

    Returns:
      counts : 1D NumPy array of dtype uint32 and length K summing to n
    """
    cdef int K = p.shape[0]
    cdef np.ndarray[np.uint32_t, ndim=1] counts_np = np.empty(K, dtype=np.uint32)
    cdef uint *counts = <uint*>counts_np.data
    cdef double total = 0.0
    cdef int i
    cdef double *probs = NULL

    # Compute total sum
    for i in range(K):
        total += p[i]
    if total <= 0.0:
        for i in range(K): counts[i] = 0
        return counts_np

    # Allocate and normalize probability buffer
    probs = <double*> PyMem_Malloc(K * sizeof(double))
    for i in range(K): probs[i] = p[i] / total

    # Sample under nogil
    with nogil:
        sample_multinomial_c(counts, K, <uint>n, probs)

    PyMem_Free(probs)
    return counts_np

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

# Column-wise sparse adjustment using nogil multinomial
def adjust_sparse_counts_cython_colwise2(object X_csr,
                                         np.ndarray[DTYPE_t, ndim=1] S):
    cdef:
        object                    X_csc, X_csc_upd
        np.ndarray[DTYPE_t, ndim=1] data_arr, new_data_arr
        np.ndarray[INTP_t, ndim=1]  indptr_arr, indices_arr
        DTYPE_t[::1]               data, new_data
        INTP_t[::1]                indptr, indices
        Py_ssize_t                 n, d, col, start, end, col_len, j, max_col_len
        long                       diff
        int                        sign
        uint                       m
        double                     total
        # reusable buffers
        np.ndarray[np.float64_t, ndim=1]  probs_arr
        np.ndarray[np.uint32_t, ndim=1]   adj_np
        double                             *probs_ptr
        uint                               *adj_ptr

    # 1) CSC + dtype views
    X_csc       = X_csr.tocsc()
    data_arr    = np.asarray(X_csc.data,   dtype=DTYPE)
    indptr_arr  = np.asarray(X_csc.indptr, dtype=np.intp)
    indices_arr = np.asarray(X_csc.indices,dtype=np.intp)
    data    = data_arr
    indptr  = indptr_arr
    indices = indices_arr
    n = X_csc.shape[0]
    d = X_csc.shape[1]

    # 2) mutable copy of data
    new_data_arr = data_arr.copy()
    new_data     = new_data_arr

    # 3) find max nnz per column
    max_col_len = 0
    for col in range(d):
        col_len = indptr[col+1] - indptr[col]
        if col_len > max_col_len:
            max_col_len = col_len

    # 4) pre-allocate buffers
    probs_arr = np.empty(max_col_len, dtype=np.float64)
    adj_np    = np.empty(max_col_len, dtype=np.uint32)
    probs_ptr = <double*>probs_arr.data
    adj_ptr   = <uint*>adj_np.data

    # 5) Column-wise pass
    for col in range(d):
        start   = indptr[col]
        end     = indptr[col+1]
        col_len = end - start
        if col_len <= 0:
            continue

        # sum + copy counts into probs
        total = 0.0
        for j in range(col_len):
            total += data[start + j]
            probs_arr[j] = data[start + j]

        # compute diff and skip
        diff = <long>total - <long>S[col]
        if diff == 0 or total <= 0.0:
            continue
        sign = -1 if diff > 0 else 1
        m = <uint>abs(diff)

        # normalize probs
        for j in range(col_len): probs_ptr[j] = probs_arr[j] / total

        # sample adjustment under nogil
        with nogil:
            sample_multinomial_c(adj_ptr, col_len, m, probs_ptr)

        # apply adjustments
        for j in range(col_len):
            c = adj_ptr[j]
            if c:
                new_data[start + j] += sign * c

    # rebuild sparse
    X_csc_upd = csc_matrix(
        (new_data_arr, indices_arr, indptr_arr),
        shape=(n, d), dtype=new_data_arr.dtype
    )
    X_csc_upd.eliminate_zeros()
    return X_csc_upd.tocsr()