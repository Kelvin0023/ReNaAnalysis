from collections import defaultdict

import mne
import numpy as np
import torch
import pickle
import os

from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset, DataLoader

from renaanalysis.learning.HT import HierarchicalTransformer, Attention
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.learning.transformer_rollout import VITAttentionRollout
from renaanalysis.params.params import random_seed, batch_size, eeg_channel_names, eeg_montage
from renaanalysis.utils.data_utils import rebalance_classes, mean_ignore_zero

def plot_training_history(history):
    # Extract the training history
    train_loss = history['loss_train']
    val_loss = history['loss_val']
    train_acc = history['acc_train']
    val_acc = history['acc_val']

    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

data_root = 'HT_grid/RSVP-itemonset-locked'
rollout_file_name = "rollout_test"
split_window_eeg = 100e-3
n_folds = 1
exg_resample_rate = 200
event_names = ["Distractor", "Target"]
head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window actiavtion
sample_fusion = 'sum'  # TODO
model_path = 'renaanalysis/learning/saved_models'
history_file_name = 'auditory_oddball_HT-pca-ica_2023-06-06_18-14-21training_histories_2.pickle'
model_file_name = 'auditory_oddball_HT-pca-ica_2023-06-06_18-14-21_2.pt'

load_saved_rollout = False


info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
info.set_montage(eeg_montage)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#Load training history
with open(os.path.join(model_path, history_file_name), 'rb') as f:
    history = pickle.load(f)

# Load x_eeg
with open(os.path.join(data_root, 'x_raw.p'), 'rb') as f:
    X = pickle.load(f)
with open(os.path.join(data_root, 'x_eeg.pkl'), 'rb') as f:
    X_pca_ica = pickle.load(f)

# Load y
with open(os.path.join(data_root, 'y_raw.p'), 'rb') as f:
    Y = pickle.load(f)
with open(os.path.join(data_root, 'y.pkl'), 'rb') as f:
    Y_pca_ica = pickle.load(f)

# Load label_encoder
with open(os.path.join(data_root, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

#plot training history
plot_training_history(history)

num_channels, num_timesteps = X_pca_ica.shape[1:]  # X is x_eeg

# Load the model state_dict
model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2, output='multi')
window_size = model.time_conv_window_size

model.load_state_dict(torch.load(os.path.join(model_path, model_file_name)))
model.to(device)
rollout = VITAttentionRollout(model, device, attention_layer_class=Attention, token_shape=model.grid_dims, discard_ratio=0.9, head_fusion=head_fusion)

skf = StratifiedShuffleSplit(n_splits=1, random_state=random_seed)
train, test = [(train, test) for train, test in skf.split(X, Y)][0]
x_eeg_train, x_eeg_pca_ica_train = X[train], X_pca_ica[train]
x_eeg_test, x_eeg_pca_ica_test = X[test], X_pca_ica[test]
y_train, y_test = Y[train], Y[test]
assert np.all(np.unique(y_test) == np.unique(y_train) ), "train and test labels are not the same"
assert len(np.unique(y_test)) == len(event_names), "number of unique labels is not the same as number of event names"
_encoder = lambda y: label_encoder.transform(y.reshape(-1, 1)).toarray()
rollout_data_root = f'HT_viz'

ht_viz(model, x_eeg_test, y_test, _encoder, event_names, rollout_data_root, model.window_duration, exg_resample_rate,
                   eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=True, head_fusion='max', discard_ratio=0.9, batch_size=64, X_pca_ica=x_eeg_pca_ica_test)

# skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed)
# for f_index, (train, test) in enumerate(skf.split(X, Y)):
#     if load_saved_rollout:
#         with open(os.path.join(data_root, f'{rollout_file_name}_{f_index}.pkl'), 'rb') as f:
#             rolls = pickle.load(f)
#         with open(os.path.join(data_root, '_x.pkl'), 'rb') as f:
#             _x = pickle.load(f)
#         with open(os.path.join(data_root, '_y.pkl'), 'rb') as f:
#             _y = pickle.load(f)
#     else:
#         x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
#
#         x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class
#
#         train_size, val_size = len(x_train), len(x_test)
#         x_train = torch.Tensor(x_train)  # transform to torch tensor
#         x_test = torch.Tensor(x_test)
#
#         y_train = torch.Tensor(y_train)
#         y_test = torch.Tensor(y_test)
#
#         train_dataset = TensorDataset(x_train, y_train)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
#
#         val_dataset = TensorDataset(x_test, y_test)
#         val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
#
#         rolls = defaultdict(list)
#         _y = []
#         _x = []
#         for i, (x, y) in enumerate(val_dataloader):
#             print(f"Rolling out attention for batch {i} of {val_size // batch_size}")
#             for j, single_x in enumerate(x): # one sample at a time
#                 print(f"Working on sample {j} of {len(x)}")
#                 for roll_depth in range(model.depth):
#                     with torch.no_grad():
#                         rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=single_x.unsqueeze(0)))
#             _y.append(y.cpu().numpy())
#             _x.append(x.cpu().numpy())
#         _x = np.concatenate(_x)
#         _y = np.concatenate(_y)
#
#         # save the rollout
#         with open(os.path.join(data_root, f'{rollout_file_name}_{f_index}.pkl'), 'wb') as f:
#             pickle.dump(rolls, f)
#         with open(os.path.join(data_root, '_x.pkl'), 'wb') as f:
#             pickle.dump(_x, f)
#         with open(os.path.join(data_root, '_y.pkl'), 'wb') as f:
#             pickle.dump(_y, f)



    # plot the topomap
    # fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    # subfigs = fig.subfigures(2, 1)
    # x_mean_max = np.max(np.mean(_x, axis=(0, -1)))
    # for class_index, e_name in enumerate(event_names):
    #     axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
    #     for window_i in range(model.num_windows):
    #         _x_class = _x[_y == class_index][:, :, (window_i) * window_size:(window_i + 1) * window_size]
    #         _x_mean = np.mean(_x_class, axis=(0, -1))
    #
    #         plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, x_mean_max))
    #         # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
    #         axes[window_i - 1].set_title(
    #             f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
    #     subfigs[class_index].suptitle(e_name, )
    # fig.suptitle(f"EEG topomap: {f_index+1}-fold", fontsize='x-large')
    # plt.show()
    # event_ids = {event_name: event_id for event_id, event_name in zip(np.sort(np.unique(Y)), event_names)}
    # _encoder = lambda y: label_encoder.transform(y.reshape(-1, 1)).toarray()
    # fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    # subfigs = fig.subfigures(2, 1)
    # x_mean_max = np.max(np.mean(_x, axis=(0, -1)))
    # for class_index, (e_name, e_id) in enumerate(event_ids.items()):
    #     axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
    #     y_event = np.squeeze(_encoder(np.array([e_id])[np.newaxis, :]))
    #     _x_class = _x[np.all(_y == y_event, axis=1)]
    #     for window_i in range(model.num_windows):
    #         _x_class_window = _x_class[:, :, (window_i) * window_size:(window_i + 1) * window_size]
    #         _x_mean = np.mean(_x_class_window, axis=(0, -1))
    #
    #         plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, x_mean_max))
    #         # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
    #         axes[window_i - 1].set_title(
    #             f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
    #     subfigs[class_index].suptitle(e_name, )
    # fig.suptitle(f"EEG topomap", fontsize='x-large')
    # plt.show()
    #
    # for roll_depth in range(model.depth):
    #     this_roll = np.stack(rolls[roll_depth], axis=0)
    #
    #     fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    #     # cross_window_activates = mean_ignore_zero(this_roll, axis=1)
    #     # cross_window_activates = np.true_divide(this_roll.sum(axis=1), (this_roll != 0).sum(axis=1))
    #     cross_window_activates = np.sum(this_roll, axis=1)  # finding max activates across channels
    #
    #     plt.boxplot(cross_window_activates)
    #     x_labels = [f"{int((i - 1) * split_window_eeg * 1e3)}ms" for i in range(model.num_windows)]
    #     x_ticks = np.arange(0.5, model.num_windows + 0.5, 1)
    #     plt.twinx()
    #     plt.plot(list(range(1, model.num_windows + 1)), np.sum(cross_window_activates, axis=0), label="Max across samples")
    #     # plt.plot(list(range(1, model.num_windows + 1)), mean_ignore_zero(cross_window_activates, axis=0), label="Max across samples")
    #
    #     plt.xticks(ticks=x_ticks, labels=x_labels)
    #     plt.xlabel("100 ms windowed bins")
    #     plt.ylabel("Cross-window attention activation")
    #     plt.title(f'Cross-window attention: {f_index+1}-fold, HT depth {roll_depth+1} of {model.depth}')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    #     fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    #     subfigs = fig.subfigures(2, 1)
    #
    #     for class_index, e_name in enumerate(event_names):
    #         axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
    #
    #         activation_max = np.max(np.sum(this_roll, axis=0))
    #         for window_i in range(model.num_windows):
    #             activation = this_roll[_y == class_index][:, :, window_i]
    #             activation = np.sum(activation, axis=0)
    #             plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, activation_max))
    #             # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
    #             axes[window_i - 1].set_title(f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
    #
    #         subfigs[class_index].suptitle(e_name)
    #     fig.suptitle(f"Attention to CLS token: {f_index+1}-fold, HT depth {roll_depth+1} of {model.depth}", fontsize='x-large')
    #     plt.show()