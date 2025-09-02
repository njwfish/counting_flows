import torch
import os
from torchvision.datasets import MNIST


class DiffMNIST:
    def __init__(self, data_dir: str = "/orcd/data/omarabu/001/njwfish/counting_flows/datasets/data/mnist"):
        self.data_dir = data_dir
        self.data_dim = 28*28
        
        # Check if preprocessed data already exists
        processed_data_path = os.path.join(data_dir, 'processed_diffmnist.pt')
        print(processed_data_path)
        print(os.path.exists(processed_data_path))
        if os.path.exists(processed_data_path):
            # Load cached preprocessed data
            print(f"Loading preprocessed MNIST data from {processed_data_path}")
            cached_data = torch.load(processed_data_path)
            self.data = cached_data['data']
            self.labels = cached_data['labels']
        else:
            # Download and preprocess data
            print(f"Downloading and preprocessing MNIST data to {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
            
            mnist = MNIST(data_dir, train=True, download=True)
            self.data = mnist.data.float()  # Convert to float for processing
            self.labels = mnist.targets
            
            # normalize data to [-1, 1]
            self.data = self.data / 255.0 * 2.0 - 1.0
            # add channel dimension
            self.data = self.data.unsqueeze(1)
            
            # Save preprocessed data for future use
            torch.save({
                'data': self.data,
                'labels': self.labels
            }, processed_data_path)
            print(f"Saved preprocessed data to {processed_data_path}")

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            'x_0': self.data[index],
            'x_1': torch.randn_like(self.data[index])
        }
