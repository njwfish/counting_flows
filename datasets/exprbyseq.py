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
        snapfile =  "/orcd/data/omarabu/001/gokul/count_sequence/training_data/onek1k_training_data/snapatac_merged_minus.h5ad",
        fastafile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/scooby_training_data/genome_human.fa",
        cell_type_col = 'cell_label',
        individual_col = 'individual',
        base_individual = '650_651',
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

        self.chrom_weights = (self.chrom_lens - window_size) / (self.chrom_lens - window_size).sum()


        self.max_nt = self.chrom_lens[-1] + self.chrom_starts[-1]
        self.n_cells = self.sd.n_obs
        self.batch_size = batch_size
        self.window_size = 196608
        
        # Cache the full sparse matrix reference
        self.fragment_matrix = self.sd.obsm['fragment_single']
        
        self.cell_type_col = cell_type_col
        self.individual_col = individual_col
        self.base_individual = base_individual
        self.base_individual_idx = np.where(self.sd.obs[individual_col] == base_individual)[0]
        self.individual_idxs = {
            individual: np.where(self.sd.obs[individual_col] == individual)[0]
            for individual in self.sd.obs[individual_col].unique() if individual != base_individual
        }

        # create one hot encoding for cell type
        target_cond = self.sd.obs[self.cell_type_col].to_numpy()
        self.target_cond = torch.zeros(len(target_cond), len(self.sd.obs[self.cell_type_col].unique()))
        for i, cell_type in enumerate(self.sd.obs[self.cell_type_col].unique()):
            self.target_cond[target_cond == cell_type, i] = 1

    def _global_to_chrom(self, ind):
        chrom_idx = (self.chrom_starts <= ind).nonzero()[0][-1]
        chrom_name = self.chrom_names[int(chrom_idx)]
        local_idx = ind - self.chrom_starts[chrom_idx]
        return chrom_name, int(local_idx)

    def get_seq(self, start: int, end: int) -> np.ndarray:
        """
        Get sequence for a given cell and window.
        """
        chrom_name, local_start = self._global_to_chrom(start)
        _, local_end = self._global_to_chrom(end)
        seq = self.fasta[chrom_name][local_start:local_end].seq
        seq = seq_to_tensor(seq)
        return seq

    def __len__(self):
        return self.n_cells // self.batch_size

    def __getitem__(self, idx):
        # random start position in genome
        chrom_idx = np.random.choice(len(self.chrom_starts), p=self.chrom_weights)
        chrom_start = self.chrom_starts[chrom_idx]
        chrom_end = chrom_start + self.chrom_lens[chrom_idx]
        start = np.random.randint(chrom_start, chrom_end - self.window_size)
        end = start + self.window_size
        seq = self.get_seq(start, end)
        
        target_start = (start + end) // 2 - self.target_size // 2
        target_end = target_start + self.target_size

        # get random set of base cells
        base_cell_idxs = np.random.choice(self.base_individual_idx, size=self.batch_size, replace=True)
        base_counts = self.fragment_matrix[base_cell_idxs][:, target_start:target_end].toarray()

        # sample random other cell type
        target_individual = np.random.choice(list(self.individual_idxs.keys()))
        target_cell_idxs = np.random.choice(self.individual_idxs[target_individual], size=self.batch_size, replace=True)
        target_counts = self.fragment_matrix[target_cell_idxs][:, target_start:target_end].toarray()

        return {
            'x_0': target_counts,
            'x_1': base_counts,
            'seq': seq,
            'class_emb': self.target_cond[target_cell_idxs]
        }

