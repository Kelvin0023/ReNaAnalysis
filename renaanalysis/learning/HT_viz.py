import os
import pickle
from collections import defaultdict
from typing import Union

import mne
import numpy as np
import torch
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from torch.utils.data import TensorDataset, DataLoader


from renaanalysis.learning.HT import HierarchicalTransformer, Attention
from renaanalysis.learning.transformer_rollout import VITAttentionRollout


def ht_viz_training(X, Y, model, roll_out_cls, encoder, device, epoch, roll_start_epoch=15):
    if epoch >= roll_start_epoch:
        Y_encoded = encoder(Y[100])
        Y_encoded_tensor = torch.Tensor(Y_encoded)
        X_tensor = torch.Tensor(X[100])

        rolls = defaultdict(list)
        for roll_depth in range(model.depth):
            with torch.no_grad():
                x_data = X_tensor.unsqueeze(0)
                # rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=x_data)
                roll = roll_out_cls(depth=roll_depth, input_tensor=x_data)
                rolls[roll_depth].append(roll)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        axes[0, 0].imshow(rolls[0][0], cmap='hot', interpolation='nearest')
        axes[0, 0].set_title(f'attention of layer {0}')
        axes[0, 1].imshow(rolls[1][0], cmap='hot', interpolation='nearest')
        axes[0, 1].set_title(f'attention of layer {1}')
        axes[1, 0].imshow(rolls[2][0], cmap='hot', interpolation='nearest')
        axes[1, 0].set_title(f'attention of layer {2}')
        axes[1, 1].imshow(rolls[3][0], cmap='hot', interpolation='nearest')
        axes[1, 1].set_title(f'attention of layer {3}')
        # plt.colorbar()
        plt.show()

def ht_viz(model: Union[str, HierarchicalTransformer], X, Y, y_encoder, event_names,
           data_root,
           split_window_eeg, exg_resample_rate, eeg_montage, num_timesteps=None, num_channels=None,
           note='',
           head_fusion='max', discard_ratio=0.1,
           load_saved_rollout=False, batch_size=64,
           X_pca_ica=None, pca=None, ica=None):
    """
    @param num_channels: number of channels for the model. This can be different from the number of channels in X. If they are different,
    we assume the model is using dimension reduced data.
    @param X_pca_ica: if None, assume model is not using dimension reduced data
    """
    event_ids = {event_name: event_id for event_id, event_name in zip(np.sort(np.unique(Y)), event_names)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(model, str):
        assert num_timesteps is not None and num_channels is not None, "Must provide num_timesteps and num_channels if model is a path"
        assert os.path.exists(model), "Model path does not exist"
        model_path = model  # save the model path
        model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    window_size = model.patch_length
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)

    rollout_fname = f'HT-rollout_{note}.pkl'
    rollout_x_fname = f'HT-rollout_x_{note}.pkl'
    rollout_y_fname = f'HT-rollout_y_{note}.pkl'

    if load_saved_rollout:
        with open(os.path.join(data_root, rollout_fname), 'rb') as f:
            rolls = pickle.load(f)
        with open(os.path.join(data_root, rollout_x_fname), 'rb') as f:
            _x = pickle.load(f)
        with open(os.path.join(data_root, rollout_y_fname), 'rb') as f:
            _y = pickle.load(f)
    else:
        # x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        # x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class

        # train_size, val_size = len(x_train), len(x_test)
        # x_train = torch.Tensor(x_train)  # transform to torch tensor
        # x_test = torch.Tensor(x_test)
        #
        # y_train = torch.Tensor(y_train)
        # y_test = torch.Tensor(y_test)

        rollout = VITAttentionRollout(model, device, attention_layer_class=Attention, token_shape=model.grid_dims, discard_ratio=discard_ratio, head_fusion=head_fusion)
        Y_encoded = y_encoder(Y)
        Y_encoded_tensor = torch.Tensor(Y_encoded).to(device)
        X_tensor = torch.Tensor(X).to(device)

        if X_pca_ica is not None:
            X_pca_ica_tensor = torch.Tensor(X_pca_ica).to(device)
        else:  # otherwise just duplicate the X_tensor
            X_pca_ica_tensor = X_tensor
        dataset = TensorDataset(X_tensor, X_pca_ica_tensor, Y_encoded_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        val_size = len(dataset)

        rolls = defaultdict(list)
        activations = defaultdict(list)
        _y = []
        _x = []
        for i, (x, x_pca_ica, y) in enumerate(dataloader):
            print(f"Rolling out attention for batch {i} of {val_size // batch_size}")
            for j, (single_x, single_x_pca_ica) in enumerate(zip(x, x_pca_ica)): # one sample at a time
                print(f"Working on sample {j} of {len(x)}")
                for roll_depth in range(model.depth):
                    with torch.no_grad():
                        x_data = single_x.unsqueeze(0) if X_pca_ica is None else single_x_pca_ica.unsqueeze(0)
                        # rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=x_data))

                        roll = rollout(depth=roll_depth, input_tensor=x_data)
                        roll_tensor = torch.Tensor(roll).to(device)
                        forward_activation = torch.empty((X.shape[1] if X_pca_ica is None else X_pca_ica.shape[1], model.num_windows, model.patch_length))
                        # if roll.shape[0] != X.shape[1]:  # HT is using dimension-reduced input

                        # compute forward activation
                        single_x_windowed = torch.chunk(single_x, model.num_windows, dim=1) if X_pca_ica is None else torch.chunk(single_x_pca_ica, model.num_windows, dim=1)
                        for window_i, x_window_data in enumerate(single_x_windowed):
                            roll_tensor_window = roll_tensor[:, window_i]
                            denom = torch.matmul(roll_tensor_window.T, roll_tensor_window)
                            if denom == 0:
                                forward_activation[:, window_i] = 0
                            else:
                                # forward_solution_pca_ica = torch.matmul(x_window_data.T, roll_tensor_window) / denom
                                forward_window = x_window_data * roll_tensor_window.view(-1, 1) / denom
                                forward_activation[:, window_i, :] = forward_window
                        if X_pca_ica is not None:
                            activation_reshaped = forward_activation.reshape((-1, model.num_windows * model.patch_length))[None, :]
                            forward_activation = pca.inverse_transform(ica.inverse_transform(activation_reshaped))[0]
                            forward_activation = forward_activation.reshape((-1, model.num_windows, model.patch_length))

                        activations[roll_depth].append(forward_activation)
                        rolls[roll_depth].append(roll)

            _y.append(y.cpu().numpy())
            _x.append(x.cpu().numpy())
        _x = np.concatenate(_x)
        _y = np.concatenate(_y)

        # save the rollout
        with open(os.path.join(data_root, rollout_fname), 'wb') as f:
            pickle.dump(rolls, f)
        with open(os.path.join(data_root, rollout_x_fname), 'wb') as f:
            pickle.dump(_x, f)
        with open(os.path.join(data_root, rollout_y_fname), 'wb') as f:
            pickle.dump(_y, f)

    # plot the topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    x_mean_max = np.max(np.mean(_x, axis=(0, -1)))
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
        _x_class = _x[np.all(_y == y_event, axis=1)]
        for window_i in range(model.num_windows):
            _x_class_window = _x_class[:, :, (window_i) * window_size:(window_i + 1) * window_size]
            _x_mean = np.mean(_x_class_window, axis=(0, -1))

            plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, x_mean_max))
            # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
            axes[window_i - 1].set_title(
                f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
        subfigs[class_index].suptitle(e_name, )
    fig.suptitle(f"EEG topomap, {note}", fontsize='x-large')
    plt.show()

    for roll_depth in range(model.depth):
        this_roll = np.stack(rolls[roll_depth], axis=0)
        this_activation = np.stack(activations[roll_depth], axis=0)

        fig = plt.figure(figsize=(15, 10))
        # cross_window_activates = mean_ignore_zero(this_roll, axis=1)
        # cross_window_activates = np.true_divide(this_roll.sum(axis=1), (this_roll != 0).sum(axis=1))
        across_channel_rolls = np.sum(this_roll, axis=1)  # finding activates across channels

        plt.boxplot(across_channel_rolls)
        x_labels = [f"{int((i - 1) * split_window_eeg * 1e3)}ms" for i in range(model.num_windows)]
        x_ticks = np.arange(0.5, model.num_windows + 0.5, 1)
        plt.twinx()
        plt.plot(list(range(1, model.num_windows + 1)), np.sum(across_channel_rolls, axis=0), label=f"Sum across samples")
        # plt.plot(list(range(1, model.num_windows + 1)), mean_ignore_zero(cross_window_activates, axis=0), label="Max across samples")

        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel("100 ms windowed bins")
        plt.ylabel("Cross-window attention activation")
        plt.title(f'Cross-window attention, {note}, HT depth {roll_depth+1} of {model.depth}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(22, 10))
        subfigs = fig.subfigures(2, 1)

        # plot the topomap rollouts
        for class_index, (e_name, e_id) in enumerate(event_ids.items()):
            axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
            y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
            activation_class = this_activation[np.all(_y == y_event, axis=1)]
            # activation_max = np.max(np.sum(activation_class, axis=0))
            # activation_min = np.min(np.sum(activation_class, axis=0))
            for window_i in range(model.num_windows):
                forward_activation = activation_class[:, :, window_i]
                forward_activation = np.sum(forward_activation, axis=0)
                forward_activation = np.mean(forward_activation, axis=1)
                # activation_max = np.max(forward_activation)

                activation_max = np.max(forward_activation, axis=0)
                activation_min = np.min(forward_activation, axis=0)

                plot_topomap(forward_activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(activation_min, activation_max))
                # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
                axes[window_i - 1].set_title(f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
            subfigs[class_index].suptitle(e_name, )

        fig.suptitle(f"Attention to the CLS token: {note}, HT depth {roll_depth+1} of {model.depth}", fontsize='x-large')
        plt.show()