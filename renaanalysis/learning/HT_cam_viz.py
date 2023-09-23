import copy
import os
import pickle
from collections import defaultdict
from typing import Union, Sequence

import mne
import numpy as np
import scipy
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mne.viz import plot_topomap

from renaanalysis.learning.cam.grad_cam import GradCAM
from renaanalysis.utils.viz_utils import get_line_styles


def ht_eeg_viz_cam(model, mmarray, attention_layer_class, device, data_root,
                                note='', head_fusion='max', discard_ratio=0.1,
                                batch_size=512,
                                use_ordered=False, eeg_montage=mne.channels.make_standard_montage('biosemi64'), topo_map='forward',
                                roll_topo_map_samples='all', roll_topo_map_n_samples=10,
                                picks=('Fz', 'Cz', 'Oz'), tmin=-0.1, tmax=0.8, *args, **kwargs):
    """
    @param model: can be the model instance or the model class. When class is provided, please give kwargs for model_init_params and model_path
    we assume the model is using dimension reduced data.
    kwargs:
    @param use_ordered: bool: whether to use the ordered test indices create from calling mmarray.training_val_test_split_ordered_by_subject_run
    if false, use the shuffled test indices generated from mmarray.training_val_test_split

    @param topo_map: str: 'forward' or 'attention', whether to plot the forward activation (eeg data scaled by attention) or just the attention
    @param roll_topo_map_samples: 'all' or 'random': if all, plot all test samples; if random, plot roll_topo_map_n_samples random samples
    @param roll_topo_map_n_samples: int: number of samples to plot if roll_topo_map_samples is 'random'
    @param picks: the eeg channel names to plot along with the topomap
    """
    assert topo_map in ['forward', 'attention'], "topo_map must be either 'forward' or 'attention'"
    assert roll_topo_map_samples in ['all', 'random'], "roll_topo_map_samples must be either 'all' or 'random'"
    assert os.path.exists(data_root), "Data root does not exist for saving rollouts"
    if isinstance(model, type):
        assert os.path.exists(kwargs['model_path']), "Model path does not exist"
        model = model(**kwargs['model_init_params'])
        model.load_state_dict(torch.load(kwargs['model_path']))
    model.to(device)
    x_test_original = mmarray['eeg'].array[mmarray.get_ordered_test_indices() if use_ordered else mmarray.test_indices]
    if use_ordered:
        test_iterator = mmarray.get_test_ordered_batch_iterator(encode_y=False,  device=device)
    else:
        test_iterator = mmarray.get_test_dataloader(batch_size=batch_size, encode_y=False, device=device)
    event_ids = mmarray.event_ids
    window_size = model.patch_length
    eeg_channel_names = eeg_montage.ch_names
    exg_resample_rate = mmarray['eeg'].sampling_rate
    split_window_eeg = model.window_duration

    info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)

    activations = dict()

    # attention rollouts
    from renaanalysis.learning.transformer_rollout_batch import VITAttentionRollout
    # rollout = VITAttentionRollout(model, device, attention_layer_class=attention_layer_class, token_shape=model.grid_dims, discard_ratio=discard_ratio, head_fusion=head_fusion)
    rollout = VITAttentionRollout(model, device, attention_layer_class=attention_layer_class, token_shape=(1, model.n_tokens), discard_ratio=discard_ratio, head_fusion=head_fusion)
    rolls = defaultdict(list)
    cams = []
    # cam from embeddings
    gradcam = GradCAM(model=model, target_layers=[model.to_patch_embedding[1]], use_cuda=torch.cuda.is_available(), reshape_transform=None)

    y_from_iterator = []
    x_eeg_from_iterator = []
    for i, (batch_data) in enumerate(test_iterator):
        y = batch_data.pop('y')
        print(f"Rolling out attention for batch {i+1} of {len(test_iterator)}")
        cams.append(gradcam(input_tensor=batch_data['eeg']))

        for roll_depth in range(model.depth):
            roll = rollout(depth=roll_depth, input_tensor=batch_data)
            roll_tensor = torch.Tensor(roll).to(device)
            rolls[roll_depth].append(roll_tensor)

        y_from_iterator.append(y)
        x_eeg_from_iterator.append(batch_data['eeg'])
    rolls = {k: torch.cat(v, dim=0) for k, v in rolls.items()}
    cams = np.concatenate(cams, axis=0)
    x_eeg_from_iterator = torch.cat(x_eeg_from_iterator, dim=0).detach().cpu().numpy()
    y_from_iterator = torch.cat(y_from_iterator, dim=0).detach().cpu().numpy()
    # compute forward activation
    cams_forward = np.multiply(cams, x_eeg_from_iterator)
    cams_forward_normalized = (cams_forward - cams_forward.mean()) / cams_forward.std()
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        cam_class = cams_forward_normalized[y_from_iterator == e_id]
        for window_i in range(model.num_windows):
            cam_class_window = cam_class[:, :, (window_i) * window_size:(window_i + 1) * window_size]
            cam_class_window = np.mean(cam_class_window, axis=0)
            cam_class_window = np.max(cam_class_window, axis=-1)
            plot_topomap(cam_class_window, info, axes=axes[window_i - 1], show=False, res=512)
            axes[window_i].set_title(f"{int((window_i * split_window_eeg + tmin) * 1e3)}-{int(((window_i+1) * split_window_eeg + tmin) * 1e3)}ms")
        subfigs[class_index].suptitle(e_name, )
    fig.suptitle(f"EEG CAM Forward, {note}", fontsize='x-large')
    plt.show()

    # plot the eeg topomap
    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    for class_index, (e_name, e_id) in enumerate(event_ids.items()):
        axes = subfigs[class_index].subplots(1, model.num_windows, sharey=True)
        # y_event = np.squeeze(y_encoder(np.array([e_id])[np.newaxis, :]))
        _x_class = x_eeg_from_iterator[y_from_iterator == e_id]
        for window_i in range(model.num_windows):
            _x_class_window = _x_class[:, :, (window_i) * window_size:(window_i + 1) * window_size]
            _x_mean = np.mean(_x_class_window, axis=(0, -1))
            plot_topomap(_x_mean, info, axes=axes[window_i - 1], show=False, res=512)
            # plot_topomap(activation, info, axes=axes[window_i - 1], show=False, res=512, vlim=(np.min(this__roll), np.max(this_roll)))
            axes[window_i].set_title( f"{int((window_i * split_window_eeg + tmin) * 1e3)}-{int(((window_i + 1) * split_window_eeg + tmin) * 1e3)}ms")
        subfigs[class_index].suptitle(e_name, )
    fig.suptitle(f"EEG topomap, {note}", fontsize='x-large')
    plt.show()

    if roll_topo_map_samples == 'random':
        roll_topo_map_pick_samples = np.random.choice(len(x_test_original), roll_topo_map_n_samples)[:, None]
    else:
        roll_topo_map_pick_samples = np.arange(len(x_test_original))[None, :]
    line_markers = get_line_styles(len(picks))
    for pick_samples in roll_topo_map_pick_samples:
        for roll_depth in range(model.depth):
            this_roll = np.stack(rolls[roll_depth], axis=0)[pick_samples]
            this_activation = activations[roll_depth][pick_samples]

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