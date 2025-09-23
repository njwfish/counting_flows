import argparse
import os
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import anndata as ad
import pandas as pd


def compute_ctp(spot_annotations, num_celltypes):
	 n_spots = len(spot_annotations)
	 ctp = np.zeros((n_spots, num_celltypes), dtype=float)
	 for i in range(n_spots):
		 labels = np.asarray(spot_annotations[i])
		 total = labels.size if hasattr(labels, 'size') else len(labels)
		 if total == 0:
			 continue
		 for cid in range(num_celltypes):
			 ctp[i, cid] = (labels == cid).sum() / float(total)
	 return ctp


def majority_vote_from_indices(indices, reference_annotations):
	 neighbor_anns = reference_annotations[indices]
	 preds = []
	 for row in neighbor_anns:
		 cnt = Counter(row)
		 preds.append(cnt.most_common(1)[0][0])
	 return np.array(preds)


def main():
	 parser = argparse.ArgumentParser(description="Generate CTP matrix from NPZ inputs following evaluation.ipynb workflow")
	 parser.add_argument("--reference", required=True, help="Path to reference S1R1 npz (with counts, annotations, spots)")
	 parser.add_argument("--predicted", required=True, help="Path to predicted counts npz (with counts, spots)")
	 parser.add_argument("--n-neighbors", type=int, default=15)
	 parser.add_argument("--batch-size", type=int, default=1000)
	 parser.add_argument("--output", help="Output CSV path for single run")
	 parser.add_argument("--output-dir", help="Output directory when running multiple cell type counts", default="/orcd/data/omarabu/001/tanush20/counting_flows_/MERFISH/outputs/")
	 parser.add_argument("--num-celltypes", type=int, help="Total number of cell types (columns) for single run")
	 parser.add_argument("--num-celltypes-list", type=str, help="Comma-separated list of cell type counts to iterate over, e.g. 5,10,15")
	 args = parser.parse_args()

	 true = np.load(args.reference, allow_pickle=True)
	 predicted = np.load(args.predicted, allow_pickle=True)

	 reference_counts = true["counts"].copy()
	 predicted_counts = predicted["counts"].copy()

	 # Preprocess reference with scanpy
	 ref_adata = ad.AnnData(reference_counts)
	 sc.pp.filter_genes(ref_adata, min_cells=3)
	 ref_adata.layers["counts"] = ref_adata.X.copy()
	 sc.pp.normalize_total(ref_adata)
	 sc.pp.log1p(ref_adata)
	 sc.tl.pca(ref_adata)
	 ref_pca = ref_adata.obsm["X_pca"]

	 # Preprocess predicted counts with the same steps (no re-fitting filtering)
	 query_adata = ad.AnnData(predicted_counts)
	 query_adata.layers["counts"] = query_adata.X.copy()
	 sc.pp.normalize_total(query_adata)
	 sc.pp.log1p(query_adata)

	 # Project predicted with PCA fitted on reference
	 # Use scanpy PCA transform by fitting sklearn PCA on ref_adata.X to align with notebook
	 from sklearn.decomposition import PCA
	 pca = PCA(n_components=ref_pca.shape[1])
	 pca.fit(ref_adata.X)
	 query_pca = pca.transform(query_adata.X)

	 # Fit NN on reference PCA and prepare neighbor indices once
	 nn_model = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean")
	 nn_model.fit(ref_pca)
	 distances, indices = nn_model.kneighbors(query_pca)

	 # Predictions will be computed below per configuration of cell types

	 # Decide single vs multiple runs
	 if args.num_celltypes_list:
		 assert args.output_dir, "--output-dir is required when using --num-celltypes-list"
		 os.makedirs(args.output_dir, exist_ok=True)
		 base_name = os.path.splitext(os.path.basename(args.reference))[0]
		 values = [int(x) for x in args.num_celltypes_list.split(',') if x.strip()]
		 for k in values:
			 reference_annotations = np.array(true["annotations"].item()[k])
			 all_preds = majority_vote_from_indices(indices, reference_annotations)
			 # Organize by spots again to keep purity per K
			 spot_annotations = []
			 j = 0
			 for arr in predicted["spots"]:
				 num = len(arr)
				 spot_annotations.append(all_preds[j:j + num])
				 j += num
			 ctp = compute_ctp(spot_annotations, k)
			 out_path = os.path.join(args.output_dir, f"S1R1_{k}/count_bridge.csv")
			 os.makedirs(os.path.dirname(out_path), exist_ok=True)
			 pd.DataFrame(ctp).to_csv(out_path)
			 print(f"Saved CTP matrix {ctp.shape} to {out_path}")
	 else:
		 assert args.num_celltypes is not None and args.output, "Provide --num-celltypes and --output for single run"
		 k = int(args.num_celltypes)
		 reference_annotations = np.array(true["annotations"].item()[k])
		 all_preds = majority_vote_from_indices(indices, reference_annotations)
		 spot_annotations = []
		 j = 0
		 for arr in predicted["spots"]:
			 num = len(arr)
			 spot_annotations.append(all_preds[j:j + num])
			 j += num
		 ctp = compute_ctp(spot_annotations, k)
		 os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
		 pd.DataFrame(ctp).to_csv(args.output)
		 print(f"Saved CTP matrix {ctp.shape} to {args.output}")


if __name__ == "__main__":
	 main()


