import numpy as np
import torch

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((0, pad_height - image.shape[0]), (0, pad_width - image.shape[1])), mode="constant")

class MerfishDeconv:
    def __init__(
        self, 
        data_dim=None, 
        data_path="/orcd/data/omarabu/001/jiaqilu/project/count_dataset/Vizgen_MERFISH/notebook/saved/", 
        num_replicates=2,
        img_size=256
    ):
        npz = np.load(data_path + "S1R1/merfish_idx.npz", allow_pickle=True)
        self.data_dim = data_dim

        self.dapi, self.counts, self.group_idxs = self.process_npz(data_path + "S1R1/merfish_idx.npz", img_size=img_size, idx_offset=0)
        offset = self.counts.shape[0]
        for replicate in range(1, num_replicates):
            dapi, counts, group_idxs = self.process_npz(data_path + f"S1R{replicate + 1}/merfish_idx.npz", img_size=img_size, idx_offset=offset)
            offset += dapi.shape[0]
            self.dapi = torch.cat([self.dapi, dapi], axis=0)
            self.counts = torch.cat([self.counts, counts], axis=0)
            self.group_idxs = np.concatenate([self.group_idxs, group_idxs], axis=0)

        
        self.size = self.group_idxs.shape[0]
        print(f"Found {self.size} groups")
        print(f"Found {self.dapi.shape[0]} images that are smaller than or equal to {img_size}")
        print("Padding images to {img_size} x {img_size}")
        print("Processing images completed")

    def process_npz(self, npz_path, img_size=256, idx_offset=0):
        npz = np.load(npz_path, allow_pickle=True)

        imgs = npz["imgs"]
        to_keep = np.array([i for i in range(imgs.shape[0]) if max(imgs[i].shape) <= img_size])
        to_keep_map = {i: j for j, i in enumerate(to_keep)}
        dapi = np.array([pad_image(imgs[i], img_size, img_size) for i in to_keep])
        tiff_max = 2**16 - 1
        dapi = dapi / tiff_max * 2.0 - 1.0
        dapi = torch.from_numpy(dapi).float()
        dapi = dapi.unsqueeze(1)

        counts = npz["counts"][to_keep]
        counts = torch.from_numpy(counts).long()

        group_idxs = npz["spots"]
        # keep only the to_keep indices within each spot
        group_idxs_to_keep = []
        for group in group_idxs:
            group_to_keep = np.array([to_keep_map[i] for i in group if i in to_keep_map]) + idx_offset
            if len(group_to_keep) > 0:
                group_idxs_to_keep.append(group_to_keep)

        group_idxs = np.array(group_idxs_to_keep, dtype=object)

        print(f"Found {len(group_idxs)} groups")
        print(f"Found {len(to_keep)} images that are smaller than or equal to {img_size}")
        return dapi, counts, group_idxs

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        group_idxs = self.group_idxs[index]
        x_0_img = self.dapi[group_idxs]
        x_0_count = self.counts[group_idxs]
        X_0 = x_0_count.sum(axis=0)

        x_1_count = torch.round(torch.abs(torch.normal(mean=0, std=10, size=x_0_count.shape))).long()
        x_1_img = torch.randn_like(self.dapi[group_idxs])

        return {
            "x_0": {
                "img":x_0_img,
                "counts": x_0_count
            },
            "x_1": {
                "img": x_1_img,
                "counts": x_1_count
            },
            "X_0": {"counts": X_0}
        }
