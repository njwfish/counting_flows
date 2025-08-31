import torch
import os
from torchvision.datasets import MNIST


class DiffMNIST:
    def __init__(self, data_dir: str = "datasets/data/mnist"):
        self.data_dir = data_dir
        mnist = MNIST(data_dir, train=True, download=True)
        self.data = mnist.data
        self.labels = mnist.targets
        self.data_dim = 28*28
        # normalize data to [-1, 1]
        self.data = self.data / 255.0 * 2.0 - 1.0
        # add channel dimension
        self.data = self.data.unsqueeze(1)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            'x_0': self.data[index],
            'x_1': torch.randn_like(self.data[index])
        }
