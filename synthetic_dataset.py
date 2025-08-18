import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern

class SyntheticMedicalDataset(Dataset):
    def __init__(self, n=200, size=64, fake_ratio=0.5):
        self.data = []
        self.labels = []
        for i in range(n):
            img = np.zeros((size, size), dtype=np.float32)
            if np.random.rand() > fake_ratio:
                rr, cc = np.ogrid[:size, :size]
                mask = (rr - size//2)**2 + (cc - size//2)**2 < (size//4)**2
                img[mask] = 1.0
                label = 0  # real
            else:
                img = np.random.rand(size, size).astype(np.float32)
                label = 1  # fake

            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            lbp = lbp / lbp.max()
            self.data.append(lbp[None, ...])  # add channel dim
            self.labels.append(label)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]
