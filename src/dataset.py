import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, in_features=384, num_classes=10):
        self.X = np.random.randn(num_samples, in_features).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]

def get_dataloader(batch_size=32):
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

