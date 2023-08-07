import pickle
import os
import warnings

import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt
from einops import rearrange, repeat
from matplotlib.colors import LinearSegmentedColormap
from mne.decoding import UnsupervisedSpatialFilter
from numpy import inf
from sklearn.decomposition import PCA
from torch import nn
import torch

from renaanalysis.learning.HT import ContrastiveLoss, MaskLayer
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.utils.utils import remove_value
from renaanalysis.learning.train import eval_test
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_samples, \
    plot_training_loss_history
from renaanalysis.params.params import *
import torch.nn.functional as F

search_params = ["lr"]
metric = 'folds val loss'
is_by_channel = False
is_pca_ica = False
is_plot_train_history = True
is_plot_ROC = False
is_plot_topomap = False
is_plot_epochs = False
is_compare_epochs = False
viz_sim = True

viz_pca_ica = False
training_histories = pickle.load(open(f'HT_grid/model_training_histories_pca_{viz_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_TUH.p', 'rb'))
locking_performance = pickle.load(open(f'HT_grid/model_locking_performances_pca_{viz_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_TUH.p', 'rb'))
models = pickle.load(open(f'HT_grid/models_with_params_pca_{viz_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_TUH.p', 'rb'))
nfolds = 1
mmarray = pickle.load(open(f'{export_data_root}/TUH_mmarray.p', 'rb'))
test_dataloader = mmarray.get_test_dataloader(batch_size=32, device='cuda:0' if torch.cuda.is_available() else 'cpu', return_metainfo=True)
is_pca_ica = 'pca' in mmarray['eeg'].data_processor.keys() or 'ica' in mmarray['eeg'].data_processor.keys()
if is_pca_ica != viz_pca_ica:
    warnings.warn('The mmarry stored is different with the one desired for visualization')
pca = mmarray['eeg'].data_processor['pca'] if 'pca' in mmarray['eeg'].data_processor.keys() else None
ica = mmarray['eeg'].data_processor['ica'] if 'ica' in mmarray['eeg'].data_processor.keys() else None
exg_resample_rate = 200
event_names = ["Distractor", "Target"]
head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window actiavtion
sample_fusion = 'sum'  # TODO


print('\n'.join([f"{str(x)}, {y[metric]}" for x, y in locking_performance.items()]))

# find the model with best test auc
# best_loss = inf
# for params, model_performance in locking_performance.items():
#     for i in range(nfolds):
#         if model_performance['folds test loss'][i] < best_loss:
#             model_idx = [params, i]
#             best_auc = model_performance['folds test auc'][i]
viz_epoch = True

# viz each layer
if viz_sim:
    for params, model_list in models.items():
        params = dict(params)
        for i in range(len(model_list)):
            for x in test_dataloader:
                model_list[i].mask_layer = MaskLayer(p_t=0.8, p_c=0.8, c_span=False, mask_t_span=1, mask_c_span=1,
                                    t_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(21, 128), requires_grad=True).to('cuda:0'),
                                    c_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(40, 128), requires_grad=True).to('cuda:0'))
                pred_series, x_encoded, mask_t, mask_c, encoder_att_matrix, decoder_att_matrix = model_list[i](*x)
                plt.rcParams["figure.figsize"] = (12.8, 7.2)
                eeg_picks = mmarray['eeg'].ch_names
                for idx, ch in enumerate(eeg_picks):
                    y = pred_series[:, idx, :].cpu().detach().numpy()
                    y_mean = np.mean(y, axis=0)
                    y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                    y2 = y_mean - scipy.stats.sem(y, axis=0)

                    time_vector = np.linspace(0, 4, y.shape[-1])
                    plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True, alpha=0.5)
                    plt.plot(time_vector, y_mean, c='red', label='{0}, N={1}'.format('Predicted', y.shape[0]))

                    y = x[0][:, idx, :1000].cpu().detach().numpy()
                    y_mean = np.mean(y, axis=0)
                    y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                    y2 = y_mean - scipy.stats.sem(y, axis=0)

                    time_vector = np.linspace(0, 4, y.shape[-1])
                    plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor='blue', interpolate=True, alpha=0.5)
                    plt.plot(time_vector, y_mean, c='blue', label='{0}, N={1}'.format('Original', y.shape[0]))

                    plt.xlabel('Time (sec)')
                    plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
                    plt.legend()

                    plt.title('{0} - Channel {1}'.format('Pretrain', ch))
                    plt.show()
                epochs.compute_psd(fmin=1, fmax=120).plot()
                for nsample, this_pred_series in enumerate(pred_series):
                    this_mask_c = repeat(mask_c[nsample][:, None], 'c 1 -> c t', t=mask_t.shape[-1])
                    this_mask_t = repeat(mask_t[nsample][None, :], '1 t -> c t', c=mask_c.shape[-1])

                    this_mask = np.logical_or(this_mask_c, this_mask_t)
                    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
                    for ch_id, time_serie in enumerate(this_pred_series):
                        axs[0].plot(time_serie.cpu().detach().numpy(), label=f'channel {ch_id}')
                        axs[0].set_title(f'predicted sample {nsample}')
                        axs[0].legend()
                        # plt.show()
                        axs[1].plot(x[0][nsample][ch_id].cpu().detach().numpy(), label=f'channel {ch_id}')
                        axs[1].set_title(f'original sample {nsample}')
                        axs[1].legend()
                    fig.show()
                    if nsample > 10:
                        break
                    # fig, axs = plt.subplots(1, 3)
                    # axs[0].imshow(this_mask)
                    # axs[0].set_title('mask')
                    # this_pred_series = rearrange(this_pred_series, 'c (x y) -> (c x) y', x=5)
                    # axs[1].imshow(this_pred_series.detach().cpu().numpy())
                    # axs[1].set_title('predicted time series')
                    # original = rearrange(x[0][nsample][:, :1000], 'c (x y) -> (c x) y', x=5)
                    # axs[2].imshow(original.detach().cpu().numpy())
                    # axs[2].set_title('original time series')
                    # plt.show()

# plot epoch
if is_plot_epochs:
    num_samp = 2
    viz_all_epoch = True
    viz_both = False
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    non_target_indc = np.where(y_test == 0)[0]
    target_indc = np.where(y_test == 6)[0]
    viz_indc = np.concatenate((non_target_indc[0:num_samp], target_indc[0:num_samp]))
    colors = {1: 'red', 7: 'blue'}
    eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names
    this_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
    model = best_model
    if viz_all_epoch:
        visualize_eeg_samples(x_test, y_test, colors, this_picks)
        ht_viz(model, x_test, y_test, _encoder, event_names, rollout_data_root,
               best_model.window_duration,
               exg_resample_rate,
               eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max',
               discard_ratio=0.9, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)
    else:
        visualize_eeg_samples(x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], colors, this_picks)
        # a = model(torch.from_numpy(x_eeg_pca_ica_test[viz_indc if viz_both else non_target_indc[0:num_samp]].astype('float32')))
        ht_viz(model, x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], _encoder, event_names, rollout_data_root, best_model.window_duration,
               exg_resample_rate,
               eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max',
               discard_ratio=0.1, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)


# plot training history
if is_plot_train_history:
    for params, history_folds in training_histories.items():
        params_dict = dict(params)
        seached_params = [params_dict[key] for key in search_params]
        for i in range(nfolds):
            history = {'loss_train': history_folds['loss_train'][i], 'loss_val': history_folds['loss_val'][i], 'loss_test': history_folds['loss_test'][i]}
            plot_training_loss_history(history, seached_params, i)


# plot ROC curve for each stored model
if is_plot_ROC:
    for params, model_list in models.items():
        for i in range(len(model_list)):
            model = model_list[i]
            test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, y_all, y_all_pred = eval_test(
                model, x_eeg_pca_ica_test, y_test, criterion, last_activation, _encoder,
                test_name='', verbose=1)
            params_dict = dict(params)
            seached_params = [params_dict[key] for key in search_params]
            viz_binary_roc(y_all, y_all_pred, seached_params, fold=i)

# plot attention weights
if is_plot_topomap:
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    ht_viz(best_model, x_test, y_test, _encoder, event_names, rollout_data_root, best_model.window_duration, exg_resample_rate,
                   eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max', discard_ratio=0.9, batch_size=64, X_pca_ica=x_eeg_pca_ica_test, pca=pca, ica=ica)


# plot performance for different params
grouped_results = {}
for key, value in locking_performance.items():
    params = [dict(key)[x] for x in search_params]
    grouped_results[tuple(params)] = value[metric]

unique_params = dict([(param_name, np.unique([key[i] for key in grouped_results.keys()])) for i, param_name in enumerate(search_params)])

# Create subplots for each parameter
fig, axes = plt.subplots(nrows=1, ncols=len(search_params), figsize=(16, 5))

# Plot the bar charts for each parameter
for i, (param_name, param_values) in enumerate(unique_params.items()):  # iterate over the hyperparameter types
    axis = axes[i]
    labels = []
    auc_values = []

    common_keys = []  # find the intersection of keys for this parameter to avoid biasing the results (needed when the grid search is incomplete)
    for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
        other_keys = [list(key) for key, value in grouped_results.items() if key[i] == param_val]
        [x.remove(param_val) for x in other_keys]
        other_keys = [tuple(x) for x in other_keys]
        common_keys.append(other_keys)
    common_keys = set(common_keys[0]).intersection(*common_keys[1:])

    for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
        # metric_values = [value for key, value in grouped_results.items() if key[i] == param_val and remove_value(key, param_val) in common_keys]
        metric_values = []
        for key, value in grouped_results.items():
            if key[i] == param_val and tuple(remove_value(key, param_val)) in common_keys:
                metric_values.append(value)

        auc_values.append(np.mean(metric_values))
        labels.append(param_val)

    xticks = np.arange(len(labels))
    axis.bar(xticks, auc_values)
    axis.set_xticks(xticks, labels=labels, rotation=45)
    axis.set_xlabel(param_name)
    axis.set_ylabel(metric)
    axis.set_title(f"HT Grid Search")

# Adjust the layout of subplots and show the figure
fig.tight_layout()
plt.show()