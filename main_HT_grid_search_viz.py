import pickle
import os

import mne
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch import nn
import torch

from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.utils.utils import remove_value
from renaanalysis.learning.train import eval
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_epoch
from renaanalysis.params.params import *

search_params = ['num_heads', 'patch_embed_dim', "feedforward_mlp_dim", "dim_head"]
metric = 'folds val auc'
is_by_channel = False
is_pca_ica = True
is_plot_train_history = False
is_plot_ROC = False
is_plot_topomap = False
is_compare_epochs = True

training_histories = pickle.load(open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'rb'))
locking_performance = pickle.load(open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'rb'))
models = pickle.load(open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'rb'))
nfolds = 3
model_dir = 'C:/Users/ixiic/PycharmProjects/ReNaAnalysis/HT_grid'
y_test = pickle.load(open(f'{export_data_root}/y_test.p', 'rb'))
y_train = pickle.load(open(f'{export_data_root}/y_train.p', 'rb'))
x_eeg_pca_ica_test = pickle.load(open(f'{export_data_root}/x_eeg_pca_ica_test.p', 'rb'))
x_eeg_test = pickle.load(open(f'{export_data_root}/x_eeg_test.p', 'rb'))
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

# find the model with best test auc
best_auc = 0
for params, model_performance in locking_performance.items():
    for i in range(nfolds):
        if model_performance['folds test auc'][i] > best_auc:
            model_idx = [params, i]
            best_auc = model_performance['folds test auc'][i]

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
            history = {'loss_train': history_folds['loss_train'][i], 'acc_train': history_folds['acc_train'][i], 'loss_val': history_folds['loss_val'][i], 'acc_val': history_folds['acc_val'][i], 'auc_val': history_folds['auc_val'][i], 'auc_test': history_folds['auc_test'][i], 'acc_test': history_folds['acc_test'][i], 'loss_test': history_folds['loss_test'][i]}
            plot_training_history(history, seached_params, i)


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
