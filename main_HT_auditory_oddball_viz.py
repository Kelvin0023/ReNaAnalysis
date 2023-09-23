from collections import defaultdict

import mne
import numpy as np
import torch
import pickle
import os

from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset, DataLoader

from renaanalysis.learning.HT import HierarchicalTransformer, Attention
from renaanalysis.learning.transformer_rollout import VITAttentionRollout
from renaanalysis.params.params import random_seed, batch_size, eeg_montage, export_data_root
from renaanalysis.utils.data_utils import rebalance_classes, mean_ignore_zero
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples

model_path = r'D:\PycharmProjects\ReNaAnalysis\renaanalysis\learning\saved_models\HT'
data_root = 'HT/auditory_oddball'

exg_resample_rate = 200
reject = 'auto'
bids_root = 'D:/Dropbox/Dropbox/ReNa/EEGDatasets/auditory_oddball_openneuro'
event_names = ["standard", "oddball_with_response"]
colors = {
    "standard": "red",
    "oddball_with_response": "green"
}
picks = 'eeg'

split_window_eeg = 100e-3
n_folds = 1
head_fusion = 'max'
channel_fusion = 'sum'  # TODO when plotting the cross window actiavtion
sample_fusion = 'sum'  # TODO
rollout_file_name = f"rollout_test_headFusion-{head_fusion}"

reload_saved_samples = True
load_saved_rollout = True


eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
info.set_montage(eeg_montage)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

X, Y, label_encoder = get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, exg_resample_rate, colors)

num_channels, num_timesteps = X.shape[1:]  # X is x_eeg

# Load the model state_dict
model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2)
window_size = model.time_conv_window_size
model.load_state_dict(torch.load(model_path))
model.to(device)
rollout = VITAttentionRollout(model, device, attention_layer_class=Attention, token_shape=model.grid_dims, discard_ratio=0.9, head_fusion=head_fusion)

skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed)


for f_index, (train, test) in enumerate(skf.split(X, Y)):
    if load_saved_rollout:
        with open(os.path.join(data_root, f'{rollout_file_name}_{f_index}.pkl'), 'rb') as f:
            rolls = pickle.load(f)
        with open(os.path.join(data_root, '_x.pkl'), 'rb') as f:
            _x = pickle.load(f)
        with open(os.path.join(data_root, '_y.pkl'), 'rb') as f:
            _y = pickle.load(f)
    else:
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

        x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class

        train_size, val_size = len(x_train), len(x_test)
        x_train = torch.Tensor(x_train)  # transform to torch tensor
        x_test = torch.Tensor(x_test)

        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)

        train_dataset = TensorDataset(x_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        val_dataset = TensorDataset(x_test, y_test)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        rolls = defaultdict(list)
        _y = []
        _x = []
        for i, (x, y) in enumerate(val_dataloader):
            print(f"Rolling out attention for batch {i} of {val_size // batch_size}")
            for j, single_x in enumerate(x): # one sample at a time
                print(f"Working on sample {j} of {len(x)}")
                for roll_depth in range(model.depth):
                    with torch.no_grad():
                        rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=single_x.unsqueeze(0)))
            _y.append(y.cpu().numpy())
            _x.append(x.cpu().numpy())
        _x = np.concatenate(_x)
        _y = np.concatenate(_y)

        # save the rollout
        with open(os.path.join(data_root, f'{rollout_file_name}_{f_index}.pkl'), 'wb') as f:
            pickle.dump(rolls, f)
        with open(os.path.join(data_root, '_x.pkl'), 'wb') as f:
            pickle.dump(_x, f)
        with open(os.path.join(data_root, '_y.pkl'), 'wb') as f:
            pickle.dump(_y, f)

    # plot the topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    x_mean_max = np.max(np.mean(_x, axis=(0, -1)))
    for class_index, e_name in enumerate(event_names):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        for window_i in range(model.num_windows):
            _x_class = _x[_y == class_index][:, :, (window_i) * window_size:(window_i + 1) * window_size]
            _x_mean = np.mean(_x_class, axis=(0, -1))

            plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, x_mean_max))
            # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
            axes[window_i - 1].set_title(
                f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
        subfigs[class_index].suptitle(e_name, )
    fig.suptitle(f"EEG topomap: {f_index+1}-fold", fontsize='x-large')
    plt.show()

    for roll_depth in range(model.depth):
        this_roll = np.stack(rolls[roll_depth], axis=0)

        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        # cross_window_activates = mean_ignore_zero(this_roll, axis=1)
        # cross_window_activates = np.true_divide(this_roll.sum(axis=1), (this_roll != 0).sum(axis=1))
        cross_window_activates = np.sum(this_roll, axis=1)  # finding max activates across channels

        plt.boxplot(cross_window_activates)
        x_labels = [f"{int((i - 1) * split_window_eeg * 1e3)}ms" for i in range(model.num_windows)]
        x_ticks = np.arange(0.5, model.num_windows + 0.5, 1)
        plt.twinx()
        plt.plot(list(range(1, model.num_windows + 1)), np.sum(cross_window_activates, axis=0), label=f"{sample_fusion} across samples")
        # plt.plot(list(range(1, model.num_windows + 1)), mean_ignore_zero(cross_window_activates, axis=0), label="Max across samples")

        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel("100 ms windowed bins")
        plt.ylabel("Cross-window attention activation")
        plt.title(f'Cross-window attention: {f_index+1}-fold, HT depth {roll_depth+1} of {model.depth}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(22, 10), constrained_layout=True)
        subfigs = fig.subfigures(2, 1)

        # plot the topomap rollouts
        for class_index, e_name in enumerate(event_names):
            axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)

            activation_max = np.max(np.sum(this_roll, axis=0))
            for window_i in range(model.num_windows):
                activation = this_roll[_y == class_index][:, :, window_i]
                # activation = np.sum(activation, axis=0)
                activation = activation[0]
                activation_max = np.max(activation)

                plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, activation_max))
                # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
                axes[window_i - 1].set_title(f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
            subfigs[class_index].suptitle(e_name, )

        fig.suptitle(f"Attention to the CLS token: {f_index+1}-fold, HT depth {roll_depth+1} of {model.depth}", fontsize='x-large')
        plt.show()