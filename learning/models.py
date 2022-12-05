import torch
import torch.nn as nn
import torch.nn.functional as F
from torchshape import tensorshape


class EEGCNNNet(nn.Module):
    def __init__(self, in_shape, num_classes, in_channels=64, num_filters=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2)
        )

        cnn_flattened_size = tensorshape(self.conv, in_shape)
        cnn_flattened_size = cnn_flattened_size[1] * cnn_flattened_size[2]

        self.fcs = nn.Sequential(
            nn.Linear(cnn_flattened_size, 128),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return x
