import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGCNNNet(nn.Module):
    def __init__(self, in_length, num_classes, in_channels=64, num_filters=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, 5),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2)
        )

        self.fcs = nn.Sequential(
            nn.Linear(8640, 128),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        x = self.fcs(x)
        return x
