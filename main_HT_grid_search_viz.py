import pickle
import os
import warnings

import mne
import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from sklearn.metrics import roc_curve, auc
from torch import nn
import torch
import torch.nn.functional as F

from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.utils.utils import remove_value
from renaanalysis.learning.train import eval_test
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_samples
from renaanalysis.params.params import *
from renaanalysis.learning.HT import HierarchicalTransformer

search_params = ['num_heads', 'patch_embed_dim', "feedforward_mlp_dim", "dim_head"]
metric = 'folds test auc'
is_by_channel = False
is_plot_train_history = False
is_plot_ROC = True
is_plot_topomap = True
is_plot_epochs = True
is_plot_pos_embedding = True

viz_pca_ica = True
training_histories = pickle.load(open(f'HT_grid/model_training_histories_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))
locking_performance = pickle.load(open(f'HT_grid/model_locking_performances_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))
models = pickle.load(open(f'HT_grid/models_with_params_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))
nfolds = 1
mmarray = pickle.load(open(f'{export_data_root}/auditory_oddball_mmarray.p', 'rb'))
x_test, y_test = mmarray.get_test_set(device='cuda:0' if torch.cuda.is_available() else 'cpu')
criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(list(models.values())[0][0], device='cuda:0' if torch.cuda.is_available() else 'cpu')
is_pca_ica = 'pca' in mmarray['eeg'].data_processor.keys() or 'ica' in mmarray['eeg'].data_processor.keys()
if is_pca_ica != viz_pca_ica:
    warnings.warn('The mmarry stored is different with the one desired for visualization')
pca = mmarray['eeg'].data_processor['pca'] if 'pca' in mmarray['eeg'].data_processor.keys() else None
ica = mmarray['eeg'].data_processor['ica'] if 'ica' in mmarray['eeg'].data_processor.keys() else None
_encoder = mmarray._encoder
exg_resample_rate = 200
event_names =  ["Distractor", "Target"]
head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window actiavtion
sample_fusion = 'sum'  # TODO


print('\n'.join([f"{str(x)}, {y[metric]}" for x, y in locking_performance.items()]))

# find the model with best test auc
best_auc = 0
for params, model_performance in locking_performance.items():
    for i in range(nfolds):
        if model_performance['folds test auc'][i] > best_auc:
            model_idx = [params, i]
            best_auc = model_performance['folds test auc'][i]

best_model = models[model_idx[0]][model_idx[1]]

# plot pos_embedding
if is_plot_pos_embedding:
    pos_embedding = best_model.pos_embedding
    pos_embedding = torch.squeeze(pos_embedding).detach().cpu()[1:]
    # pos_embedding = rearrange(pos_embedding[1:], '(c t) (h w) -> (c h) (w t)', c=64, t=9, h=16, w=8)
    # plt.imshow(pos_embedding[0:64], cmap='viridis')
    # plt.colorbar()
    # plt.title(f'pos_embedding')
    # plt.show()

    pos_embedding = pos_embedding.reshape((64, 9, -1))
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(eeg_channel_names, sfreq=exg_resample_rate, ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)
    # for row in pos_embedding:
    #     for column in row:
    #         tiled_vector = column.unsqueeze(0).unsqueeze(0).repeat(64, 9, 1)
    #         sim = F.cosine_similarity(tiled_vector, pos_embedding, dim=-1)
    #         sim_array = sim.numpy()
    #         # plt.imshow(sim_array, cmap='viridis')
    #         # plt.colorbar()
    #         # plt.title(f'pos_embedding')
    #         # plt.show()
    #         fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    #         axes = fig.subplots(1, 9, sharey=True)
    #         for window in range(9):
    #             plot_topomap(sim_array[:, window], info, axes=axes[window], show=False, res=512,
    #                      vlim=(np.min(sim_array[:, window]), np.max(sim_array[:, window])))
    #             axes[window].set_title(
    #                 f"{int((window) * best_model.window_duration * 1e3)}-{int((window+1) * best_model.window_duration * 1e3)}ms")
    #
    #         fig.suptitle(f"EEG topomap", fontsize='x-large')
    #         plt.show()
    for row in pos_embedding:
        fig = plt.figure(figsize=(22, 10), constrained_layout=True)
        axes = fig.subplots(1, 9, sharey=True)
        for idx, column in enumerate(row):
            tiled_vector = column.unsqueeze(0).unsqueeze(0).repeat(64, 9, 1)
            sim = F.cosine_similarity(tiled_vector, pos_embedding, dim=-1)
            sim_array = sim.numpy()
            # plt.imshow(sim_array, cmap='viridis')
            # plt.colorbar()
            # plt.title(f'pos_embedding')
            # plt.show()
            # for window in range(9):
            #     plot_topomap(sim_array[:, window], info, axes=axes[window], show=False, res=512,
            #              vlim=(np.min(sim_array[:, window]), np.max(sim_array[:, window])))
            #     axes[window].set_title(
            #         f"{int((window) * best_model.window_duration * 1e3)}-{int((window+1) * best_model.window_duration * 1e3)}ms")
            plot_topomap(np.sum(sim_array, axis=1), info, axes=axes[idx], show=False, res=512,
                         vlim=(np.min(np.sum(sim_array, axis=1)), np.max(np.sum(sim_array, axis=1))))

        fig.suptitle(f"EEG topomap", fontsize='x-large')
        plt.show()

# plot epoch
if is_plot_epochs:
    num_samp = 2
    viz_all_epoch = True
    viz_both = False
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_test.shape[1:]
    non_target_indc = np.where(y_test == 0)[0]
    target_indc = np.where(y_test == 6)[0]
    viz_indc = np.concatenate((non_target_indc[0:num_samp], target_indc[0:num_samp]))
    colors = {0: 'red', 6: 'blue'}
    eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names
    this_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
    if viz_all_epoch:
        visualize_eeg_samples(x_test, y_test, colors, this_picks)
        x_test = x_test[:128]
        y_test = y_test[:128]
        ht_viz(model, x_test, y_test, _encoder, event_names, rollout_data_root,
               best_model.window_duration,
               exg_resample_rate,
               eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=True, head_fusion='max',
               discard_ratio=0.9, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)
    else:
        visualize_eeg_samples(x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], colors, this_picks)
        # a = model(torch.from_numpy(x_eeg_pca_ica_test[viz_indc if viz_both else non_target_indc[0:num_samp]].astype('float32')))
        ht_viz(model, x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], _encoder, event_names, rollout_data_root, best_model.window_duration,
               exg_resample_rate,
               eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=True, head_fusion='max',
               discard_ratio=0.1, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)


# plot training history
if is_plot_train_history:
    for params, history_folds in training_histories.items():
        params_dict = dict(params)
        seached_params = [params_dict[key] for key in search_params]
        for i in range(nfolds):
            history = {'loss_train': history_folds['loss_train'][i], 'acc_train': history_folds['acc_train'][i], 'loss_val': history_folds['loss_val'][i], 'acc_val': history_folds['acc_val'][i], 'auc_val': history_folds['auc_val'][i], 'auc_test': history_folds['auc_test'][i], 'acc_test': history_folds['acc_test'][i], 'loss_test': history_folds['loss_test'][i]}
            plot_training_history(history, seached_params, i)


# plot ROC curve for each stored model
if is_plot_ROC:
    for params, model_list in models.items():
        for i in range(len(model_list)):
            model = model_list[i]
            new_model = HierarchicalTransformer(180, 20, 200, num_classes=2,
                                                extraction_layers=None,
                                                depth=params['depth'], num_heads=params['num_heads'],
                                                feedforward_mlp_dim=params['feedforward_mlp_dim'],
                                                pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
                                                dim_head=params['dim_head'], emb_dropout=params['emb_dropout'],
                                                attn_dropout=params['attn_dropout'], output=params['output'],
                                                training_mode='classification')
            model = new_model.load_state_dict(model.state_dict())
            test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, y_all, y_all_pred = eval(
                model, x_eeg_pca_ica_test, y_test, criterion, last_activation, _encoder,
                test_name=TaskName.Normal.value, verbose=1)
            params_dict = dict(params)
            seached_params = [params_dict[key] for key in search_params]
            viz_binary_roc(y_all, y_all_pred, seached_params, fold=i)

# plot attention weights
if is_plot_topomap:
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_eeg_pca_ica_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    ht_viz(best_model, x_eeg_test, y_test, _encoder, event_names, rollout_data_root, best_model.window_duration, exg_resample_rate,
                   eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max', discard_ratio=0.9, batch_size=64, X_pca_ica=x_eeg_pca_ica_test, pca=pca, ica=ica)


# plot performance for different params
grouped_results = {}
for key, value in locking_performance.items():
    params = [dict(key)[x] for x in search_params]
    grouped_results[tuple(params)] = sum(value[metric]) / len(value[metric])

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
