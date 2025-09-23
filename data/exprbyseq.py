import snapatac2_scooby as snap
import numpy as np
from pyfaidx import Fasta
import pyranges as pr


class scExprBySeq:
    def __init__(self, 
                snapfile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/epicardioids_training_data/snapatac_matched_minus.h5ad", 
                fastafile = "/orcd/data/omarabu/001/gokul/count_sequence/training_data/scooby_training_data/genome_human.fa",
                gtffile="/orcd/data/omarabu/001/gokul/counting_flows/data/gencode.v43.annotation.gtf"):
        self.sd = snap.read(snapfile)
        self.fasta = Fasta(fastafile)
        self.chrom_lens = np.array(self.sd.uns['reference_sequences']['reference_seq_length'])
        self.chrom_names = self.sd.uns['reference_sequences']['reference_seq_name']
        self.chrom_starts = np.hstack(([0], self.chrom_lens.cumsum()[:-1]))
        self.chrom_map = {name: i for i, name in enumerate(self.chrom_names)}
        self.max_nt = self.chrom_lens[-1] + self.chrom_starts[-1]
        self.n_cells = self.sd.n_obs

        self.annotations = pr.read_gtf(gtffile)
        # Filter for just gene entries for faster lookups
        self.genes = self.annotations[self.annotations.Feature == 'gene'].as_df()

    def _global_to_chrom(self, ind):
        chrom_idx = (self.chrom_starts <= ind).nonzero()[0][-1]
        chrom_name = self.chrom_names[int(chrom_idx)]
        local_idx = ind - self.chrom_starts[chrom_idx]
        return chrom_name, int(local_idx)

    def get_window_by_gene(self, gene_name: str):
        """
        args:
            gene_name (str): gene name (e.g. 'SOX2'). case-sensitive.
            
        returns:
            (global_start, global_end).
        """
        # find the gene in the annotations DataFrame
        gene_info = self.genes[self.genes.gene_name == gene_name]

        if gene_info.empty:
            raise ValueError(f"{gene_name} not found in the GTF file.")
        
        # take the first entry if there are multiple matches
        gene = gene_info.iloc[0]
        
        chrom = gene.Chromosome
        start = gene.Start
        end = gene.End
                    
        # sad and ugly :( but functional 
        chrom_idx = self.chrom_map[chrom]
        global_start = self.chrom_starts[chrom_idx] + start
        global_end = self.chrom_starts[chrom_idx] + end
        
        return (int(global_start), int(global_end))

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