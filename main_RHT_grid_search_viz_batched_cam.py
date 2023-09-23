import pickle
import time
import warnings

import torch
from matplotlib import pyplot as plt

from renaanalysis.learning.HT import HierarchicalTransformer, Attention
from renaanalysis.learning.HT_cam_viz import ht_eeg_viz_cam
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.params.params import *
from renaanalysis.utils.utils import remove_value
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_samples

search_params = ['num_heads', 'patch_embed_dim', "feedforward_mlp_dim", "dim_head"]
metric = 'folds test auc'
is_by_channel = False
is_plot_train_history = False
is_plot_ROC = True
is_plot_topomap = True
is_plot_epochs = True
device ='cuda:0' if torch.cuda.is_available() else 'cpu'
nfolds = 1

use_ordered = True
viz_pca_ica = False
dataset_name = 'auditory_oddball'
result_root = 'RHT_grid_search'
colors = {1: 'red', 7: 'blue'}

########################################################################################################################

mmarray = pickle.load(open(f'{export_data_root}/{dataset_name}_mmarray_class-weight.p', 'rb'))

training_histories = pickle.load(open(f'{result_root}/{dataset_name}/model_training_histories_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))
locking_performance = pickle.load(open(f'{result_root}/{dataset_name}/model_locking_performances_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))
models = pickle.load(open(f'{result_root}/{dataset_name}/models_with_params_pca_{viz_pca_ica}_chan_{is_by_channel}.p', 'rb'))

criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(list(models.values())[0][0], device='cuda:0' if torch.cuda.is_available() else 'cpu')
if use_ordered:
    test_data = mmarray.get_ordered_test_set(device=device, convert_to_tensor=False)
else:
    test_data = mmarray.get_test_set(device=device, convert_to_tensor=False)

x_eeg_test, y_test = test_data['eeg'], test_data['y']
# get the un-pca-ica version of the data
x_test_original = mmarray['eeg'].array[mmarray.get_ordered_test_indices() if use_ordered else mmarray.test_indices]

is_pca_ica = 'pca' in mmarray['eeg'].data_processor.keys() or 'ica' in mmarray['eeg'].data_processor.keys()
if is_pca_ica != viz_pca_ica:
    warnings.warn('The mmarry stored is different with the one desired for visualization')
pca = mmarray['eeg'].data_processor['pca'] if 'pca' in mmarray['eeg'].data_processor.keys() else None
ica = mmarray['eeg'].data_processor['ica'] if 'ica' in mmarray['eeg'].data_processor.keys() else None
_encoder = mmarray._encoder
exg_resample_rate = 200
event_names =  ["Distractor", "Target"]
head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window activation
sample_fusion = 'sum'  # TODO
discard_ratio = 0.9


print('\n'.join([f"{str(x)}, {y[metric]}" for x, y in locking_performance.items()]))

# find the model with best test auc
best_auc = 0
for params, model_performance in locking_performance.items():
    for i in range(nfolds):
        if model_performance['folds test auc'][i] > best_auc:
            model_idx = [params, i]
            best_auc = model_performance['folds test auc'][i]

# plot epoch
if is_plot_epochs:
    num_samp = 2
    viz_all_epoch = True
    viz_both = False
    rollout_data_root = f'HT_viz'
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
    num_channels, num_timesteps = x_eeg_test.shape[1:]
    best_model = models[model_idx[0]][model_idx[1]]
    non_target_indc = np.where(y_test == 0)[0]
    target_indc = np.where(y_test == 6)[0]
    viz_indc = np.concatenate((non_target_indc[0:num_samp], target_indc[0:num_samp]))
    eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names
    this_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
    model = best_model
    if viz_all_epoch:
        visualize_eeg_samples(x_eeg_test, y_test, colors, this_picks)
        t_start = time.perf_counter()
        ht_eeg_viz_cam(best_model, mmarray, Attention, device, rollout_data_root,
                              note='', load_saved_rollout=False, head_fusion=head_fusion, cls_colors=mmarray.event_viz_colors,
                              discard_ratio=discard_ratio, is_pca_ica=is_pca_ica, pca=pca, ica=ica, batch_size=16, use_ordered=use_ordered)
        print("ht viz batched took {} seconds".format(time.perf_counter() - t_start))
    else:
        visualize_eeg_samples(x_eeg_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], colors, this_picks)
        ht_eeg_viz_cam(model, x_eeg_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], _encoder, event_names, rollout_data_root, best_model.window_duration,
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
