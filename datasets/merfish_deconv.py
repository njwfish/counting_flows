import numpy as np
import torch

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((0, pad_height - image.shape[0]), (0, pad_width - image.shape[1])), mode="constant")

class MerfishDeconv:
    def __init__(self, data_dim=None,data_path="/orcd/data/omarabu/001/njwfish/counting_flows/datasets/data/merfish.npz", size=50_000):
        npz = np.load(data_path, allow_pickle=True)
        self.data_dim = data_dim
        self.size = size

        print(f"Filtering out images that are larger than 256")
        # filter out images that are larger than 256
        imgs = npz["imgs"]
        to_keep = np.array([i for i in range(imgs.shape[0]) if max(imgs[i].shape) <= 256])
        
        print(f"Found {len(to_keep)} images that are smaller than or equal to 256")
        print("Padding images to 256x256")

        self.dapi = np.array([pad_image(imgs[i], 256, 256) for i in to_keep])
        self.counts = npz["counts"][to_keep]

        # normalize dapi to [-1, 1] using
        tiff_max = 2**16 - 1
        self.dapi = self.dapi / tiff_max * 2.0 - 1.0

        # cast to torch 
        self.dapi = torch.from_numpy(self.dapi).float()
        self.dapi = self.dapi.unsqueeze(1)
        self.counts = torch.from_numpy(self.counts).long()

        print("Processing images completed")


        self.size = size
        base_size = self.counts.shape[0]
        group_sizes = np.round(np.abs(np.random.normal(loc=15, scale=20, size=size))).astype(int) + 1
        # sample 1000 groups of size group_size from base_size indices
        self.group_idxs = np.array([np.random.randint(0, base_size, size=group_size) for group_size in group_sizes], dtype=object)

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
