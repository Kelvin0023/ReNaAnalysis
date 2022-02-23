"""https://github.com/pytorch/examples/blob/master/word_language_model/"""

import math
import torch
import torch.nn as nn

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device="cuda"):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])

        return out

    def predict(self, input):
        """Makes prediction for the set of inputs provided and returns the same
        Args:
            input ([torch.Tensor]): [A tensor of inputs]
        """
        with torch.no_grad():
            predictions = self.forward(input)
        return predictions
