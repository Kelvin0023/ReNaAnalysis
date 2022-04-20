import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset
import torch.nn.functional as F

from learning_utils import interpolate_epochs_nan


class EpochDatset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, label_path, transform=None, device="cuda"):
        """
        Args:

        """
        self.X = interpolate_epochs_nan(np.load(data_path))  # remove nan
        assert np.sum(np.isnan(self.X)) == 0

        self.X = np.mean(self.X, axis=1)
        self.X = (self.X - np.mean(self.X)) / np.std(self.X)  # z-norm x

        self.y = np.load(label_path)
        self.num_classes = len(np.unique(self.y))

        oversample = SMOTE()
        self.X, self.y = oversample.fit_resample(self.X, self.y)
        self.X = self.X[:, :, None]  # add one dimension
                                       # onehot encode y
        self.X = torch.tensor(self.X, dtype=torch.float, device=device)  # average left and right eye
        self.y = torch.tensor(self.y, dtype=torch.int64, device=device)
        self.y = F.one_hot(self.y - 1, num_classes=self.num_classes).to(dtype=float)  # minus one because our smallest label (distractor) number starts at 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]