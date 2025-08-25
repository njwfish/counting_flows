import snapatac2_scooby as snap
import numpy as np
from pyfaidx import Fasta


class scExprBySeq:
    def __init__(self, 
                snapfile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/epicardioids_training_data/snapatac_matched_minus.h5ad", 
                fastafile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/scooby_training_data/genome_human.fa"):
        self.sd = snap.read(snapfile)
        self.fasta = Fasta(fastafile)
        self.chrom_lens = np.array(self.sd.uns['reference_sequences']['reference_seq_length'])
        self.chrom_names = self.sd.uns['reference_sequences']['reference_seq_name']
        self.chrom_starts = np.hstack(([0], self.chrom_lens.cumsum()[:-1]))
        self.max_nt = self.chrom_lens[-1] + self.chrom_starts[-1]
        self.n_cells = self.sd.n_obs

    def _global_to_chrom(self, ind):
        chrom_idx = (self.chrom_starts <= ind).nonzero()[0][-1]
        chrom_name = self.chrom_names[int(chrom_idx)]
        local_idx = ind - self.chrom_starts[chrom_idx]
        return chrom_name, int(local_idx)

    def get(self, cell_idx, window):

        assert max(cell_idx) < self.n_cells, "cell index bad"
        assert min(cell_idx) > -1, "cell index bad"
        assert min(window) > -1, "window index bad"
        assert max(window) < self.max_nt, "window index bad"

        start, end = window
        length = end - start

        x = self.sd.obsm['fragment_single'][cell_idx][:, start:end]

        count_vecs = []
        for i in range(x.shape[0]):
            vec = np.zeros(length)
            start_ptr, end_ptr = x.indptr[i], x.indptr[i+1]
            cols = x.indices[start_ptr:end_ptr]
            vals = x.data[start_ptr:end_ptr]
            for col, val in zip(cols, vals):
                vec[col+val:col] += 1
            count_vecs.append(vec)

        chrom_name, local_start = self._global_to_chrom(start)
        _, local_end = self._global_to_chrom(end)
        seq = self.fasta[chrom_name][local_start:local_end].seq

        return count_vecs, seq