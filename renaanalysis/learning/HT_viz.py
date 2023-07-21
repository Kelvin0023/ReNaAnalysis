import os
import pickle
from collections import defaultdict
from typing import Union

import mne
import numpy as np
import torch
from einops import rearrange
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
           is_pca_ica=False, pca=None, ica=None, attention_layer_class=Attention, X_original=None):
    """
    @param num_channels: number of channels for the model. This can be different from the number of channels in X. If they are different,
    we assume the model is using dimension reduced data.
    @param X_pca_ica: if None, assume model is not using dimension reduced data
    """
    if not is_pca_ica:
        X_original = X
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

    rollout_fname = f'HT-rollout_{note}_.pkl'
    activation_fname = f'HT-activation_{note}_.pkl'
    rollout_x_fname = f'HT-rollout_x_{note}_.pkl'
    rollout_y_fname = f'HT-rollout_y_{note}_.pkl'

    if load_saved_rollout:
        with open(os.path.join(data_root, activation_fname), 'rb') as f:
            activations = pickle.load(f)
        with open(os.path.join(data_root, rollout_fname), 'rb') as f:
            rolls = pickle.load(f)
        with open(os.path.join(data_root, rollout_x_fname), 'rb') as f:
            _x = pickle.load(f)
        with open(os.path.join(data_root, rollout_y_fname), 'rb') as f:
            _y = pickle.load(f)
    else:
        activations = defaultdict(list)

        rollout = VITAttentionRollout(model, device, attention_layer_class=attention_layer_class, token_shape=model.grid_dims, discard_ratio=discard_ratio, head_fusion=head_fusion)
        Y_encoded = y_encoder(Y)
        Y_encoded_tensor = torch.Tensor(Y_encoded).to(device)
        X_tensor = torch.Tensor(X).to(device)

        dataset = TensorDataset(X_tensor, Y_encoded_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        val_size = len(dataset)

        rolls = defaultdict(list)
        _y = []
        _x = []
        for i, (x, y) in enumerate(dataloader):
            print(f"Rolling out attention for batch {i} of {val_size // batch_size}")
            for j, single_x in enumerate(x): # one sample at a time
                print(f"Working on sample {j} of {len(x)}")
                for roll_depth in range(model.depth):
                    with torch.no_grad():
                        x_data = single_x.unsqueeze(0)
                        # rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=x_data))

                        roll = rollout(depth=roll_depth, input_tensor=x_data)
                        roll_tensor = torch.Tensor(roll).to(device)
                        forward_activation = torch.empty((X.shape[1], model.num_windows, model.patch_length))
                        # if roll.shape[0] != X.shape[1]:  # HT is using dimension-reduced input

                        # compute forward activation
                        single_x_windowed = torch.chunk(single_x, model.num_windows, dim=1)
                        for window_i, x_window_data in enumerate(single_x_windowed):
                            roll_tensor_window = roll_tensor[:, window_i]
                            denom = torch.matmul(roll_tensor_window.T, roll_tensor_window)
                            if denom == 0:
                                forward_activation[:, window_i] = 0
                            else:
                                # forward_solution_pca_ica = torch.matmul(x_window_data.T, roll_tensor_window) / denom
                                forward_window = x_window_data * roll_tensor_window.view(-1, 1) / denom
                                forward_activation[:, window_i, :] = forward_window
                        if is_pca_ica:
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
        with open(os.path.join(data_root, activation_fname), 'wb') as f:
            pickle.dump(activations, f)
        with open(os.path.join(data_root, rollout_fname), 'wb') as f:
            pickle.dump(rolls, f)
        with open(os.path.join(data_root, rollout_x_fname), 'wb') as f:
            pickle.dump(_x, f)
        with open(os.path.join(data_root, rollout_y_fname), 'wb') as f:
            pickle.dump(_y, f)

    # plot the topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    x_mean_max = np.max(np.mean(X_original, axis=(0, -1)))
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
        _x_class = X_original[np.all(_y == y_event, axis=1)]
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


def ht_viz_multimodal(model, mmarray,
           data_root,
           note='',
           head_fusion='max', discard_ratio=0.1,
           load_saved_rollout=False, batch_size=64,
           is_pca_ica=False, pca=None, ica=None, attention_layer_class=Attention, use_meta_info=False, *args, **kwargs):
    """
    @param num_channels: number of channels for the model. This can be different from the number of channels in X. If they are different,
    we assume the model is using dimension reduced data.
    @param X_pca_ica: if None, assume model is not using dimension reduced data
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(model, type):
        assert os.path.exists(kwargs['model_path']), "Model path does not exist"
        model = model(**kwargs['model_init_params'])
        model.load_state_dict(torch.load(kwargs['model_path']))
    test_iterator = mmarray.get_test_dataloader(batch_size=batch_size, encode_y=False, return_metainfo=use_meta_info,device=device)
    x_eeg_test, y_test = test_iterator.dataset.tensors[0].detach().cpu().numpy(), test_iterator.dataset.tensors[-1].detach().cpu().numpy()

    model.to(device)
    window_size = model.patch_length
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = eeg_montage.ch_names
    n_chan = len(eeg_channel_names)
    exg_resample_rate = mmarray['eeg'].sampling_rate
    info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * n_chan)
    info.set_montage(eeg_montage)

    split_window_eeg = model.window_duration

    event_ids = mmarray.event_ids
    y_encoder = mmarray._encoder

    rollout_fname = f'HT-rollout_{note}_.pkl'
    activation_fname = f'HT-activation_{note}_.pkl'
    rollout_x_fname = f'HT-rollout_x_{note}_.pkl'
    rollout_y_fname = f'HT-rollout_y_{note}_.pkl'

    if load_saved_rollout:
        with open(os.path.join(data_root, activation_fname), 'rb') as f:
            activations = pickle.load(f)
        with open(os.path.join(data_root, rollout_fname), 'rb') as f:
            rolls = pickle.load(f)
        with open(os.path.join(data_root, rollout_x_fname), 'rb') as f:
            x_eeg_from_iterator = pickle.load(f)
        with open(os.path.join(data_root, rollout_y_fname), 'rb') as f:
            y_from_iterator = pickle.load(f)
    else:
        activations = defaultdict(list)

        rollout = VITAttentionRollout(model, device, attention_layer_class=attention_layer_class, token_shape=model.grid_dims, discard_ratio=discard_ratio, head_fusion=head_fusion)

        rolls = defaultdict(list)
        y_from_iterator = []
        x_eeg_from_iterator = []
        for i, batch_data in enumerate(test_iterator):
            y = batch_data[-1]
            x = batch_data[:-1]
            print(f"Rolling out attention for batch {i} of {len(test_iterator)}")
            for j, single_x in enumerate(zip(*x)): # one sample at a time
                print(f"Working on sample {j} of {len(x[0])}")
                for roll_depth in range(model.depth):
                    with torch.no_grad():
                        x_data = tuple([s_x.unsqueeze(0) for s_x in single_x])
                        # rolls[roll_depth].append(rollout(depth=roll_depth, input_tensor=x_data))

                        roll = rollout(depth=roll_depth, input_tensor=x_data)
                        roll_tensor = torch.Tensor(roll).to(device)
                        forward_activation = torch.empty((n_chan, model.num_windows, model.patch_length))
                        # if roll.shape[0] != X.shape[1]:  # HT is using dimension-reduced input

                        # compute forward activation
                        x_eeg = single_x[0]
                        single_x_windowed = torch.chunk(x_eeg, model.num_windows, dim=1)
                        for window_i, x_window_data in enumerate(single_x_windowed):
                            roll_tensor_window = roll_tensor[:, window_i]
                            denom = torch.matmul(roll_tensor_window.T, roll_tensor_window)
                            if denom == 0:
                                forward_activation[:, window_i] = 0
                            else:
                                # forward_solution_pca_ica = torch.matmul(x_window_data.T, roll_tensor_window) / denom
                                forward_window = x_window_data * roll_tensor_window.view(-1, 1) / denom
                                forward_activation[:, window_i, :] = forward_window
                        if is_pca_ica:
                            activation_reshaped = forward_activation.reshape((-1, model.num_windows * model.patch_length))[None, :]
                            forward_activation = pca.inverse_transform(ica.inverse_transform(activation_reshaped))[0]
                            forward_activation = forward_activation.reshape((-1, model.num_windows, model.patch_length))

                        activations[roll_depth].append(forward_activation.cpu().numpy())
                        rolls[roll_depth].append(roll)
                # if j == 2:
                #     break

            y_from_iterator.append(y.cpu().numpy())
            x_eeg_from_iterator.append(x[0].cpu().numpy())
            # if i == 2:
            #     break
        x_eeg_from_iterator = np.concatenate(x_eeg_from_iterator)
        y_from_iterator = np.concatenate(y_from_iterator)

        # save the rollout
        with open(os.path.join(data_root, activation_fname), 'wb') as f:
            pickle.dump(activations, f)
        with open(os.path.join(data_root, rollout_fname), 'wb') as f:
            pickle.dump(rolls, f)
        with open(os.path.join(data_root, rollout_x_fname), 'wb') as f:
            pickle.dump(x_eeg_from_iterator, f)
        with open(os.path.join(data_root, rollout_y_fname), 'wb') as f:
            pickle.dump(y_from_iterator, f)

    # plot the topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    x_mean_max = np.max(np.mean(x_eeg_from_iterator, axis=(0, -1)))
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
        _x_class = x_eeg_from_iterator[np.all(y_from_iterator == y_event, axis=1)]
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
            activation_class = this_activation[np.all(y_from_iterator == y_event, axis=1)]
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


def ht_eeg_viz_multimodal_batch(model, mmarray, attention_layer_class, device, data_root,
                                note='', head_fusion='max', discard_ratio=0.1,
                                load_saved_rollout=False, batch_size=512,
                                is_pca_ica=False, pca=None, ica=None, use_meta_info=False, *args, **kwargs):
    """
    @param model: can be the model instance or the model class. When class is provided, please give kwargs for model_init_params and model_path
    @param num_channels: number of channels for the model. This can be different from the number of channels in X. If they are different,
    we assume the model is using dimension reduced data.
    @param X_pca_ica: if None, assume model is not using dimension reduced data

    kwargs:
    @param model_path
    @param model_init_params
    """

    if isinstance(model, type):
        assert os.path.exists(kwargs['model_path']), "Model path does not exist"
        model = model(**kwargs['model_init_params'])
        model.load_state_dict(torch.load(kwargs['model_path']))
    x_test_original = mmarray['eeg'].array[mmarray.test_indices]
    test_iterator = mmarray.get_test_dataloader(batch_size=batch_size, encode_y=False, return_metainfo=use_meta_info,device=device)
    n_samples = len(test_iterator.dataset)
    event_ids = mmarray.event_ids
    model.to(device)
    window_size = model.patch_length
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = eeg_montage.ch_names
    n_eeg_chan = len(eeg_channel_names)
    n_model_chan = model.num_channels
    exg_resample_rate = mmarray['eeg'].sampling_rate
    split_window_eeg = model.window_duration
    y_encoder = mmarray._encoder

    info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)

    rollout_fname = f'HT-rollout_{note}_.pkl'
    activation_fname = f'HT-activation_{note}_.pkl'
    rollout_x_fname = f'HT-rollout_x_{note}_.pkl'
    rollout_y_fname = f'HT-rollout_y_{note}_.pkl'

    if load_saved_rollout:
        with open(os.path.join(data_root, activation_fname), 'rb') as f:
            activations = pickle.load(f)
        with open(os.path.join(data_root, rollout_fname), 'rb') as f:
            rolls = pickle.load(f)
        with open(os.path.join(data_root, rollout_x_fname), 'rb') as f:
            x_eeg_from_iterator = pickle.load(f)
        with open(os.path.join(data_root, rollout_y_fname), 'rb') as f:
            y_from_iterator = pickle.load(f)
    else:
        activations = dict()

        from renaanalysis.learning.transformer_rollout_batch import VITAttentionRollout
        rollout = VITAttentionRollout(model, device, attention_layer_class=attention_layer_class, token_shape=model.grid_dims, discard_ratio=discard_ratio, head_fusion=head_fusion)
        rolls = defaultdict(list)
        y_from_iterator = []
        x_eeg_from_iterator = []
        for i, (batch_data) in enumerate(test_iterator):
            y = batch_data[-1]
            batch_inputs = batch_data[:-1]
            print(f"Rolling out attention for batch {i+1} of {len(test_iterator)}")
            for roll_depth in range(model.depth):
                with torch.no_grad():
                    roll = rollout(depth=roll_depth, input_tensor=batch_inputs)
                    roll_tensor = torch.Tensor(roll).to(device)
                rolls[roll_depth].append(roll_tensor)

            y_from_iterator.append(y)
            x_eeg_from_iterator.append(batch_inputs[0])
        rolls = {k: torch.cat(v, dim=0) for k, v in rolls.items()}

        # compute forward activation
        x_eeg_from_iterator = torch.cat(x_eeg_from_iterator, dim=0)
        x_eeg_windowed = torch.chunk(x_eeg_from_iterator.to(device), model.num_windows, dim=-1)
        x_eeg_from_iterator = x_eeg_from_iterator.cpu().numpy()
        y_from_iterator = torch.cat(y_from_iterator, dim=0).cpu().numpy()

        for roll_depth in range(model.depth):
            forward_activation_windowed = torch.empty((n_samples, n_model_chan, model.num_windows, model.patch_length)).to(device)
            for window_i, x_window_data in enumerate(x_eeg_windowed):
                roll_tensor_window = rolls[roll_depth][:, :, window_i]
                denom = torch.einsum('b c, b c -> b', roll_tensor_window, roll_tensor_window)
                forward_activation_windowed[denom==0, :, window_i, :] = 0
                forward_window = torch.einsum('b c t, b c -> b c t', x_window_data, roll_tensor_window) / denom[:, None, None].repeat(1, n_model_chan, model.patch_length)
                forward_activation_windowed[:, :, window_i] = forward_window
            if is_pca_ica:
                forward_activation = rearrange(forward_activation_windowed, 'b c w t -> b c (w t)')   # concat windows for inverse transform
                forward_activation_inversed = pca.inverse_transform(ica.inverse_transform(forward_activation.detach().cpu().numpy()))
                forward_activation_windowed = rearrange(torch.Tensor(forward_activation_inversed).to(device), 'b c (w t) -> b c w t', w=model.num_windows, t=model.patch_length)  # split windows again

            activations[roll_depth] = forward_activation_windowed.detach().cpu().numpy()
        rolls = {k: v.detach().cpu().numpy() for k, v in rolls.items()}
        # save the rollout
        with open(os.path.join(data_root, activation_fname), 'wb') as f:
            pickle.dump(activations, f)
        with open(os.path.join(data_root, rollout_fname), 'wb') as f:
            pickle.dump(rolls, f)
        with open(os.path.join(data_root, rollout_x_fname), 'wb') as f:
            pickle.dump(x_eeg_from_iterator, f)
        with open(os.path.join(data_root, rollout_y_fname), 'wb') as f:
            pickle.dump(y_from_iterator, f)

    # plot the topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    x_mean_max = np.max(np.mean(x_test_original, axis=(0, -1)))
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        # y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
        _x_class = x_test_original[y_from_iterator == e_id]
        for window_i in range(model.num_windows):
            _x_class_window = _x_class[:, :, (window_i) * window_size:(window_i + 1) * window_size]
            _x_mean = np.mean(_x_class_window, axis=(0, -1))

            plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512, vlim=(0, x_mean_max))
            # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
            axes[window_i - 1].set_title(f"{int((window_i - 1) * split_window_eeg * 1e3)}-{int(window_i * split_window_eeg * 1e3)}ms")
        subfigs[class_index].suptitle(e_name, )
    fig.suptitle(f"EEG topomap, {note}", fontsize='x-large')
    plt.show()

    for roll_depth in range(model.depth):
        this_roll = np.stack(rolls[roll_depth], axis=0)
        this_activation = activations[roll_depth]

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
            # y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
            activation_class = this_activation[y_from_iterator == e_id]
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