import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGInceptionNet(nn.Module):
    def __init__(self, in_shape, num_classes, in_channels=1, num_filters=8):
        super().__init__()
        self.incep_0_0 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, (1, 64), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, (8, 1), padding='valid'),  # spatial conv
            nn.BatchNorm2d(num_filters),
        )
        self.incep_0_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, (1, 32), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, (8, 1), padding='valid'),  # spatial conv
            nn.BatchNorm2d(num_filters),
        )
        self.incep_0_2 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, (1, 16), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, (8, 1), padding='valid'),  # spatial conv
            nn.BatchNorm2d(num_filters),
        )

        self.incep_1_0 = nn.Sequential(
            nn.Conv2d(3 * num_filters, num_filters, (1, 16), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
        )
        self.incep_1_1 = nn.Sequential(
            nn.Conv2d(3 * num_filters, num_filters, (1, 8), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
        )
        self.incep_1_2 = nn.Sequential(
            nn.Conv2d(3 * num_filters, num_filters, (1, 4), padding='same'),  # temporal conv
            nn.BatchNorm2d(num_filters),
        )

        self.incep_0_pool = nn.Sequential(
            nn.AvgPool2d((1, 4))
        )

        self.conv_head = nn.Sequential(
            nn.AvgPool2d((1, 2)),
            nn.Conv2d(3 * num_filters, 12, (1, 8), padding='same'),  # temporal conv
            nn.AvgPool2d((1, 2)),
            nn.Conv2d(12, 6, (1, 4), padding='same'),  # temporal conv
            nn.AvgPool2d((1, 2)),
            nn.Flatten()
        )

        if len(in_shape) == 3:
            in_shape = list(in_shape)
            in_shape.insert(1, 1)

        # with torch.no_grad():
        #     cnn_flattened_size = self.conv(torch.rand(in_shape)).shape[1]

        self.fcs = nn.Sequential(
            # nn.Linear(9976, 128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            nn.Linear(1368, num_classes)
        )

    def forward(self, input):
        x = torch.concat([self.incep_0_0(input), self.incep_0_1(input), self.incep_0_2(input)], dim=1)
        x = self.incep_0_pool(x)
        x = torch.concat([self.incep_1_0(x), self.incep_1_1(x), self.incep_1_2(x)], dim=1)
        x = self.conv_head(x)

        x = self.fcs(x)
        return x

    def prepare_data(self, x):
        if len(x.shape) == 3:
            return x[:, None, :, :]


class EEGCNN(nn.Module):
    def __init__(self, in_shape, num_classes, num_filters=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_shape[1], num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # with torch.no_grad():
        #     cnn_flattened_size = self.conv(torch.rand(in_shape)).shape[1]

        self.fcs = nn.Sequential(
            nn.Linear(304, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.fcs(x)
        return x

    def prepare_data(self, x):
        return x

    def forward_without_classification(self, input):
        x = self.conv(input)
        for fc in self.fcs[:-1]:
            x = fc(x)
        return x

class EEGPupilCNN(nn.Module):
    def __init__(self, eeg_in_shape, pupil_in_shape, num_classes, pupil_in_channel=2, num_filters=16, fc_feature_size=432):
        super().__init__()
        self.conv_eeg = nn.Sequential(
            nn.Conv1d(eeg_in_shape[1], num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        self.conv_pupil = nn.Sequential(
            nn.Conv1d(pupil_in_channel, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 3),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # with torch.no_grad():
        #     cnn_flattened_size = self.conv(torch.rand(in_shape)).shape[1]

        self.fcs = nn.Sequential(
            nn.Linear(fc_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, input):
        x_eeg = self.conv_eeg(input[0])
        x_pupil = self.conv_pupil(input[1])
        x = torch.concat([x_eeg, x_pupil], dim=1)
        x = self.fcs(x)
        return x

    def prepare_data(self, x):
        return x