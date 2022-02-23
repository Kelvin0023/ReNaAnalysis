import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import interpolate_epochs_nan


class EpochDatset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, label_path, transform=None, device="cuda"):
        """
        Args:

        """

        self.X = interpolate_epochs_nan(np.load(data_path))  # average left and right eye
        self.X = np.mean(self.X, axis=1)
        self.X = torch.tensor(self.X, dtype=torch.float, device=device)  # average left and right eye
        assert torch.sum(torch.isnan(self.X)) == 0
        # z-norm x
        self.X = (self.X - torch.mean(self.X)) / torch.std(self.X)

        self.X = self.X[:, :, None]  # add one dimension
        self.y = torch.tensor(np.load(label_path), dtype=torch.int64, device=device)
        self.num_classes = len(torch.unique(self.y))
        # onehot encode y
        self.y = F.one_hot(self.y - 1, num_classes=self.num_classes).to(dtype=float)  # minus one because our smallest label (distractor) number starts at 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]