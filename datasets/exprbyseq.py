import snapatac2_scooby as snap
import numpy as np
from pyfaidx import Fasta
from scipy.sparse import csr_matrix, coo_matrix
from typing import Union, List, Tuple


import torch
import numpy as np

# make translation table
table = str.maketrans("ACGTNacgtn", "0123401234")  # dummy digits for mapping

def seq_to_tensor(seq: str) -> torch.Tensor:
    # translate to digits string, then view as bytes
    digits = seq.translate(table).encode("ascii")
    arr = np.frombuffer(digits, dtype=np.uint8) - ord('0')
    return arr.astype(int)

class ExprBySeq:
    """
    Ultra-efficient vectorized version of scExprBySeq that completely avoids Python loops.
    
    This version uses the most efficient approach:
    1. Direct sparse matrix operations without any loops
    2. Vectorized range creation using advanced numpy techniques
    3. Optimal memory usage by keeping everything sparse
    """
    
    def __init__(
        self, 
        snapfile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/epicardioids_training_data/snapatac_matched_minus.h5ad", 
        fastafile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/scooby_training_data/genome_human.fa",
        cell_type_col = 'clusters',
        base_cell_type = '9',
        window_size = 1000,
        batch_size = 100,
        min_value = 0,
    ):
        self.data_dim = 196608
        self.target_size = 896
        self.sd = snap.read(snapfile)
        self.fasta = Fasta(fastafile)
        self.chrom_lens = np.array(self.sd.uns['reference_sequences']['reference_seq_length'])
        self.chrom_names = self.sd.uns['reference_sequences']['reference_seq_name']
        self.chrom_starts = np.hstack(([0], self.chrom_lens.cumsum()[:-1]))
        self.max_nt = self.chrom_lens[-1] + self.chrom_starts[-1]
        self.n_cells = self.sd.n_obs
        self.batch_size = batch_size
        self.window_size = window_size
        
        # Cache the full sparse matrix reference
        self.fragment_matrix = self.sd.obsm['fragment_single']

        self.base_cell_type = base_cell_type
        self.base_cell_type_idx = np.where(self.sd.obs[cell_type_col] == base_cell_type)[0]
        self.cell_type_idxs = {
            cell_type: np.where(self.sd.obs[cell_type_col] == cell_type)[0]
            for cell_type in self.sd.obs[cell_type_col].unique() if cell_type != base_cell_type
        }

    def _global_to_chrom(self, ind: Union[int, np.ndarray]) -> Union[Tuple[str, int], Tuple[List[str], np.ndarray]]:
        """Optimized chromosome lookup using searchsorted"""
        if np.isscalar(ind):
            chrom_idx = np.searchsorted(self.chrom_starts[1:], ind, side='right')
            chrom_name = self.chrom_names[int(chrom_idx)]
            local_idx = ind - self.chrom_starts[int(chrom_idx)]
            return chrom_name, int(local_idx)
        else:
            chrom_indices = np.searchsorted(self.chrom_starts[1:], ind, side='right')
            chrom_names = [self.chrom_names[i] for i in chrom_indices]
            local_indices = ind - self.chrom_starts[chrom_indices]
            return chrom_names, local_indices

    def get_sparse(self, cell_idx: Union[int, List[int], np.ndarray], 
                   window: Tuple[int, int]) -> Tuple[csr_matrix, str]:
        """
        Get count data as sparse matrix using completely vectorized operations.
        
        This is the most efficient method - no Python loops at all.
        """
        # Ensure cell_idx is array-like
        if np.isscalar(cell_idx):
            cell_idx = [cell_idx]
        cell_idx = np.asarray(cell_idx, dtype=np.int32)
        
        # Validate inputs
        assert np.max(cell_idx) < self.n_cells, "cell index bad"
        assert np.min(cell_idx) >= 0, "cell index bad"
        assert window[0] >= 0, "window index bad"
        assert window[1] <= self.max_nt, "window index bad"

        start, end = window
        length = end - start
        
        # Extract the sparse submatrix for the requested cells and window
        x_sparse = self.fragment_matrix[cell_idx][:, start:end]
        
        # Convert to COO format for efficient manipulation
        x_coo = x_sparse.tocoo()
        
        if x_coo.nnz == 0:
            # No fragments in this region
            count_matrix = csr_matrix((len(cell_idx), length), dtype=np.int32)
        else:
            # The key insight: completely vectorized conversion
            count_matrix = self._fragments_to_counts_vectorized(x_coo, len(cell_idx), length)
        
        return count_matrix

    def get_seq(self, start: int, end: int) -> np.ndarray:
        """
        Get sequence for a given cell and window.
        """
        chrom_name, local_start = self._global_to_chrom(start)
        _, local_end = self._global_to_chrom(end)
        seq = self.fasta[chrom_name][local_start:local_end].seq
        seq = seq_to_tensor(seq)
        return seq

    def _fragments_to_counts_vectorized(self, fragment_coo: coo_matrix, n_cells: int, n_positions: int) -> csr_matrix:
        """
        Ultra-efficient vectorized conversion using only numpy operations.
        
        This implements the original logic: vec[col+val:col] += 1
        where val can be negative (fragment extends backwards from col).
        """
        if fragment_coo.nnz == 0:
            return csr_matrix((n_cells, n_positions), dtype=np.int32)
        
        cell_indices = fragment_coo.row
        fragment_positions = fragment_coo.col  # This is the 'col' in the original code
        fragment_vals = fragment_coo.data      # This is the 'val' in the original code
        
        # Original logic: vec[col+val:col] += 1
        # When val is negative, this creates coverage from (col+val) to col
        coverage_starts = fragment_positions + fragment_vals  # col + val
        coverage_ends = fragment_positions                     # col
        
        # Handle directionality: ensure start <= end
        actual_starts = np.minimum(coverage_starts, coverage_ends)
        actual_ends = np.maximum(coverage_starts, coverage_ends)
        
        # Clip to valid range [0, n_positions)
        actual_starts = np.clip(actual_starts, 0, n_positions)
        actual_ends = np.clip(actual_ends, 0, n_positions)
        
        # Calculate coverage lengths
        coverage_lengths = actual_ends - actual_starts
        
        # Keep only fragments with positive coverage
        valid_mask = coverage_lengths > 0
        if not np.any(valid_mask):
            return csr_matrix((n_cells, n_positions), dtype=np.int32)
        
        # Filter to valid fragments
        cell_indices = cell_indices[valid_mask]
        actual_starts = actual_starts[valid_mask]
        actual_ends = actual_ends[valid_mask]
        coverage_lengths = coverage_lengths[valid_mask]
        
        
        # Total number of positions we need to create
        total_positions = np.sum(coverage_lengths)
        
        if total_positions == 0:
            return csr_matrix((n_cells, n_positions), dtype=np.int32)
        
        # Create row indices (which cell each position belongs to)
        output_rows = np.repeat(cell_indices, coverage_lengths)
    
        # Method: Use broadcasting and advanced indexing
        # 1. Create a base array for all the ranges we need
        max_range_length = np.max(coverage_lengths)
        
        # 2. Create indices for each range using broadcasting
        # This creates a 2D array where each row contains indices for one range
        range_indices = np.arange(max_range_length)[None, :] < coverage_lengths[:, None]
        
        # 3. Create the actual column values using broadcasting
        base_positions = actual_starts[:, None]  # Shape: (n_fragments, 1)
        offset_positions = np.arange(max_range_length)[None, :]  # Shape: (1, max_length)
        
        # 4. Generate all possible positions
        all_positions = base_positions + offset_positions  # Shape: (n_fragments, max_length)
        
        # 5. Extract only the valid positions using the mask
        output_cols = all_positions[range_indices].astype(np.int32)
        
        # Create values (all ones)
        output_vals = np.ones(total_positions, dtype=np.int32)
        
        # Build the sparse matrix
        count_matrix = csr_matrix(
            (output_vals, (output_rows, output_cols)), 
            shape=(n_cells, n_positions), 
            dtype=np.int32
        )
        
        # Sum duplicates (multiple fragments covering the same position)
        count_matrix.sum_duplicates()
        
        return count_matrix

    def __len__(self):
        return self.n_cells // self.batch_size

    def __getitem__(self, idx):
        # random start position in genome
        start = np.random.randint(0, self.max_nt - self.window_size)
        end = start + self.window_size
        seq = self.get_seq(start, end)
        target_start = (start + end) // 2 - self.target_size // 2
        target_end = target_start + self.target_size

        # get random set of base cells
        base_cell_idxs = np.random.choice(self.base_cell_type_idx, size=self.batch_size, replace=True)
        base_counts = self.get_sparse(base_cell_idxs, (target_start, target_end)).toarray()

        # sample random other cell type
        target_cell_type = np.random.choice(list(self.cell_type_idxs.keys()))
        target_cell_idxs = np.random.choice(self.cell_type_idxs[target_cell_type], size=self.batch_size, replace=True)
        target_counts = self.get_sparse(target_cell_idxs, (target_start, target_end)).toarray()

        return {
            'x_0': target_counts,
            'x_1': base_counts,
            'z': seq
        }

