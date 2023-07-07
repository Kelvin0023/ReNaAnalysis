import pickle
import os

import mne
import numpy as np
from matplotlib import pyplot as plt
from einops import rearrange, repeat
from matplotlib.colors import LinearSegmentedColormap
from torch import nn
import torch

from renaanalysis.learning.HT import ContrastiveLoss
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.utils.utils import remove_value
from renaanalysis.learning.train import eval
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_epoch, \
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
is_compare_epochs = False
viz_sim = True

training_histories = pickle.load(open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_bendr.p', 'rb'))
locking_performance = pickle.load(open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_bendr.p', 'rb'))
models = pickle.load(open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_bendr.p', 'rb'))
nfolds = 1
y_test = pickle.load(open(f'{export_data_root}/y_test.p', 'rb'))
y_train = pickle.load(open(f'{export_data_root}/y_train.p', 'rb'))
x_eeg_pca_ica_test = pickle.load(open(f'{export_data_root}/x_eeg_pca_ica_test.p', 'rb'))
x_eeg_test = pickle.load(open(f'{export_data_root}/x_eeg_test.p', 'rb'))
x_eeg_test = np.asarray(x_eeg_test, dtype='float32')
label_encoder = pickle.load(open(f'{export_data_root}/label_encoder.p', 'rb'))
if is_pca_ica:
    pca = pickle.load(open(f'{export_data_root}/pca_object.p', 'rb'))
    ica = pickle.load(open(f'{export_data_root}/ica_object.p', 'rb'))
criterion = nn.CrossEntropyLoss()
last_activation = nn.Sigmoid()
_encoder = lambda y: label_encoder.transform(y.reshape(-1, 1)).toarray()
exg_resample_rate = 200
event_names = ["Distractor", "Target"]
head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window actiavtion
sample_fusion = 'sum'  # TODO


print('\n'.join([f"{str(x)}, {y[metric]}" for x, y in locking_performance.items()]))

# viz each layer
if viz_sim:
    loss = ContrastiveLoss(0.1, 20)
    for params, model_list in models.items():
        for i in range(len(model_list)):
            device = 'cuda:0' if model_list[i].to_patch_embedding[1].bias.is_cuda else 'cpu'
            x = model_list[i].to_patch_embedding(torch.from_numpy(x_eeg_test[0:10]).to(device))
            if model_list[i].training_mode == 'self-sup pretrain':
                x, original_x, mask_t, mask_c = model_list[i].mask_layer(x)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

            b, n, _ = x.shape

            cls_tokens = repeat(model_list[i].cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += model_list[i].pos_embedding[:, :(n + 1)]
            x = model_list[i].dropout(x)

            x, att_matrix = model_list[i].transformer(x)
            x = model_list[i].to_latent(x[:, 1:].transpose(1, 2).view(original_x.shape))
            sim = F.cosine_similarity(x.permute(0, 2, 3, 1), original_x.permute(0, 2, 3, 1), dim=-1)
            sim_array = sim.cpu().detach().numpy()
            for idx, epoch in enumerate(sim_array):
                cmap_colors = [(0.0, 'blue'), (1.0, 'red')]
                custom_cmap = LinearSegmentedColormap.from_list('CustomColormap', cmap_colors)
                plt.imshow(epoch, cmap=custom_cmap)
                plt.colorbar()
                plt.title(f'Predicted Simularity of epoch {idx}')
                plt.show()
            # loss._calculate_similarity(original_x, x)
# find the model with best test auc
best_loss = 0
for params, model_performance in locking_performance.items():
    for i in range(nfolds):
        if model_performance['folds test loss'][i] > best_loss:
            model_idx = [params, i]
            best_loss = model_performance['folds test loss'][i]

# plot epoch
if is_compare_epochs:
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_eeg_pca_ica_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    ht_viz(best_model, x_eeg_test[0:10], y_test[0: 10], _encoder, event_names, rollout_data_root, best_model.window_duration,
           exg_resample_rate,
           eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max',
           discard_ratio=0.9, batch_size=64, X_pca_ica=x_eeg_pca_ica_test, pca=pca, ica=ica)


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
            test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, y_all, y_all_pred = eval(
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
    num_channels, num_timesteps = x_eeg_pca_ica_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    ht_viz(best_model, x_eeg_test, y_test, _encoder, event_names, rollout_data_root, best_model.window_duration, exg_resample_rate,
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