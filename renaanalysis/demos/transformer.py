import os
import pickle
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import mne
import numpy as np
from imblearn.over_sampling import SMOTE
from mne.viz import plot_topomap
from scipy import stats
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})


data_x_path = 'C:/Data/x_auditory_oddball.p'
data_y_path = 'C:/Data/y_auditory_oddball.p'

with open(data_x_path, 'rb') as f:
    data_x = pickle.load(f)
with open(data_y_path, 'rb') as f:
    data_y = pickle.load(f)

print(f"{data_x.shape = }")
print(f"{data_y.shape = }")

for class_label in np.unique(data_y):
    print(f"{class_label = } {np.sum(data_y == class_label) = }")

def z_norm_by_trial(data):
    """
    Z-normalize data by trial, the input data is in the shape of (num_samples, num_channels, num_timesteps)
    @param data: data is in the shape of (num_samples, num_channels, num_timesteps)
    """
    norm_data = np.copy(data)
    for i in range(data.shape[0]):
        sample = data[i]
        mean = np.mean(sample, axis=(0, 1))
        std = np.std(sample, axis=(0, 1))
        sample_norm = (sample - mean) / std
        norm_data[i] = sample_norm
    return norm_data

data_x = z_norm_by_trial(data_x)


# rebalance our data
def rebalance_classes(x, y, by_channel=False, random_seed=None):
    """
    Resamples the data to balance the classes using SMOTE algorithm.

    Parameters:
        x (np.ndarray): Input data array of shape (epochs, channels, samples).
        y (np.ndarray): Target labels array of shape (epochs,).
        by_channel (bool): If True, balance the classes separately for each channel. Otherwise,
            balance the classes for the whole input data.

    Returns:
        tuple: A tuple containing the resampled input data and target labels as numpy arrays.
    """
    epoch_shape = x.shape[1:]

    if by_channel:
        y_resample = None
        channel_data = []
        channel_num = epoch_shape[0]

        # Loop through each channel and balance the classes separately
        for channel_index in range(0, channel_num):
            sm = SMOTE(random_state=random_seed)
            x_channel = x[:, channel_index, :]
            x_channel, y_resample = sm.fit_resample(x_channel, y)
            channel_data.append(x_channel)

        # Expand dimensions for each channel array and concatenate along the channel axis
        channel_data = [np.expand_dims(x, axis=1) for x in channel_data]
        x = np.concatenate([x for x in channel_data], axis=1)
        y = y_resample

    else:
        # Reshape the input data to 2D array and balance the classes
        x = np.reshape(x, newshape=(len(x), -1))
        sm = SMOTE(random_state=random_seed)
        x, y = sm.fit_resample(x, y)

        # Reshape the input data back to its original shape
        x = np.reshape(x, newshape=(len(x),) + epoch_shape)

    return x, y

data_x, data_y = rebalance_classes(data_x, data_y, by_channel=False, random_seed=42)

for class_label in np.unique(data_y):
    print(f"{class_label = } {np.sum(data_y == class_label) = }")

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attention = self.attend(dots)
        attention = self.dropout(attention)  # TODO

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attention


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, feedforward_mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, feedforward_mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x)
            x = out + x
            x = prenorm_feedforward(x) + x
        return x, attention  # last layer


class PhysioTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.5, emb_dropout=0.5, output='multi'):
        """

        # a token is a time slice of data on a single channel

        @param num_timesteps: int: number of timesteps in each sample
        @param num_channels: int: number of channels of the input data
        @param output: str: can be 'single' or 'multi'. If 'single', the output is a single number to be put with sigmoid activation. If 'multi', the output is a vector of size num_classes to be put with softmax activation.
        note that 'single' only works when the number of classes is 2.
        """
        if output == 'single':
            assert num_classes == 2, 'output can only be single when num_classes is 2'
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.window_duration = window_duration

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.path_embed_dim = patch_embed_dim
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length

        self.grid_dims = self.num_channels, self.num_windows
        self.num_patches = self.num_channels * self.num_windows

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, attn_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, num_classes))

    def forward(self, x_eeg):
            x = self.encode(x_eeg)
            return self.mlp_head(x)

    def encode(self, x_eeg):
        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, att_matrix = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

label_onehot_encoder = preprocessing.OneHotEncoder()
y_encoded = label_onehot_encoder.fit_transform(data_y.reshape(-1, 1)).toarray()

print(f"{np.unique(data_y) = }")
print(f"{np.unique(y_encoded, axis=0) = }")

_, num_channels, num_timesteps = data_x.shape
sampling_rate = 200
model = PhysioTransformer(num_timesteps=num_timesteps, num_channels=num_channels, sampling_rate=sampling_rate, num_classes=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

# Set device

# Split the data into training and validation sets
x_train, x_val, y_train_encoded, y_val_encoded = train_test_split(data_x, y_encoded, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded)

# Convert the data into PyTorch tensors
x_train = torch.Tensor(x_train)
x_val = torch.Tensor(x_val)
y_train_encoded = torch.Tensor(y_train_encoded)
y_val_encoded = torch.Tensor(y_val_encoded)

# Define hyperparameters
batch_size = 32
lr = 0.001
num_epochs = 100

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
last_activation = nn.Softmax(dim=1)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = TensorDataset(x_train, y_train_encoded)
val_dataset = TensorDataset(x_val, y_val_encoded)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = []
    num_correct_preds = 0
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs} training', unit="batch")
    for x, y in train_dataloader:
        y_tensor = y.to(device)
        # Forward pass
        outputs = model(x.to(device))
        y_pred = last_activation(outputs)
        loss = criterion(y_pred, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

        # Calculate training predictions for accuracy calculation
        preds = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

        predicted_labels = torch.argmax(y_pred, dim=1)
        true_label = torch.argmax(y_tensor, dim=1)
        num_correct_preds += torch.sum(true_label == predicted_labels).item()

        # Update progress bar description with loss
        pbar.update(1)
        pbar.set_postfix(loss=loss.item())
    pbar.close()

    # Compute the average training loss for the epoch
    avg_loss = np.mean(running_loss)

    # Calculate training accuracy
    train_acc = num_correct_preds / len(train_dataset)

    # Evaluate on the validation set
    model.eval()
    val_running_loss = []
    num_correct_preds = 0
    val_preds = []
    with torch.no_grad():
        pbar = tqdm(total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs} validating', unit="batch")
        for x, y in val_dataloader:
            y_tensor = y.to(device)
            # Forward pass
            outputs = model(x.to(device))
            y_pred = last_activation(outputs)
            loss = criterion(y_pred, y_tensor)

            val_running_loss.append(loss.item())

            # Calculate training predictions for accuracy calculation
            preds = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

            predicted_labels = torch.argmax(y_pred, dim=1)
            true_label = torch.argmax(y_tensor, dim=1)
            num_correct_preds += torch.sum(true_label == predicted_labels).item()

            # Update progress bar description with loss
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
    pbar.close()

    # Compute the average validation loss
    val_avg_loss = np.mean(val_running_loss)
    # Calculate validation accuracy
    val_acc = num_correct_preds / len(val_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.4f}")

print("Training finished!")