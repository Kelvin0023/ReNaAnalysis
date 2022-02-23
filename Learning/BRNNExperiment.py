import math

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import numpy as np

# load data
from Learning.BRNN import BRNN
from Learning.EpochDataset import EpochDatset

data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP.npy'
label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epoch_labels_pupil_raw_condition_RSVP.npy'
batch_size = 16
train_test_split_ratio = 0.8
num_layers = 3
hidden_size = 8

learning_rate = 1e-3
num_epochs = 10
device = "cuda"

epoch_dataset = EpochDatset(data_path, label_path)
val_N = math.ceil(len(epoch_dataset) * train_test_split_ratio)
train_N = len(epoch_dataset) - val_N
val, train = random_split(epoch_dataset, [train_N, val_N])

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True)

input_size = epoch_dataset[0][0].size()[-1]  # get sequence length
num_classes = epoch_dataset.num_classes

BRNN_model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(BRNN_model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print("Epoch {0} of {1}".format(epoch + 1, num_epochs))
    for batch_idx, (batch, y_expected) in enumerate(train_loader):
        print("On {0} of {1} batch".format(batch_idx, batch_size), end='\r', flush=True)
        # Get data to cuda if possible
        y_expected = y_expected.to(dtype=torch.float)

        # forward
        assert torch.sum(torch.isnan(batch)) == 0
        y_predicted = BRNN_model(batch)
        loss = loss_function(y_predicted, y_expected)
        print("On {0} of {1} batch, loss={2}".format(batch_idx, batch_size, loss), end='\r', flush=True)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    num_correct = 0
    validation_loss = 0

    with torch.no_grad():
        for x_sample, y_sample in val_loader:
            y_sample_pred = BRNN_model(x_sample)
            validation_loss += (loss_function(y_sample_pred, y_sample).item() / x_sample.shape[0])
            num_correct += (torch.argmax(y_sample_pred, dim=-1) == torch.argmax(y_sample, dim=-1)).sum()
        print("Val loss is {0}, accuracy is {1}".format(validation_loss, num_correct/len(val)))
        # print(
        #     f"Got {num_correct} / {num_samples} with accuracy  \
        #       {float(num_correct) / float(num_samples) * 100:.2f}"
        # )