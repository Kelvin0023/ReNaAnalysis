import os.path
import pickle

from matplotlib import pyplot as plt
import torch

from renaanalysis.learning.HT_viz import ht_viz, ht_eeg_viz_multimodal_batch, ht_viz_multimodal
from renaanalysis.learning.RHT import RelAttention
from renaanalysis.multimodal.multimodal import load_mmarray
from renaanalysis.utils.utils import remove_value
from renaanalysis.utils.viz_utils import viz_binary_roc, plot_training_history, visualize_eeg_samples
from renaanalysis.params.params import *
from renaanalysis.learning.HT import HierarchicalTransformer

results_path = r'D:\PycharmProjects\ReNaAnalysis\RHT_grid_search_simEEGfMRIPupil'

# viz options
plot_eeg_epochs = False
plot_ht_viz = True

#
viz_search_param_axes = ['num_heads', 'patch_embed_dim', "feedforward_mlp_dim", "dim_head"]  # the axes of search parameters to be visualized
metric = 'folds test auc'
is_plot_topomap = True

# for visualizing eeg epochs
this_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']

# parameters for attention rollout
head_fusion = 'mean'
channel_fusion = 'sum'
sample_fusion = 'sum'
discard_ratio = 0.9
num_sample = 2
rollout_data_root = f'RHT_viz'
attention_layer_class = RelAttention
load_saved_rollout = False
use_ordered = True

mmarray_path = os.path.join(results_path, 'mmarray.p')
"""
model_param_dict has keys of model parameters combination as string. They can be cast into dict using json.loads
each value is list of three things 
* index of the param combination among all combinations, the index can used to retrieve model of the corresponding param 
combination. Models saved from RHT grid search only has this index in its file name because the limitation on how long
a file name can be.
* param_performance: a dict of performance metrics, with keys of 'folds val auc', 'folds val acc', 'folds train acc', 
'folds val loss', 'folds train loss', 'folds test auc'. Each with only one value, the result of the best fold for this
param combination. The most important value is 'folds test auc', which should be used to select the best model.

"""
model_param_dict_path = os.path.join(results_path, 'model_param_dict.p')

# End of user parameters ###############################################################################################
device ='cuda:0' if torch.cuda.is_available() else 'cpu'
mmarray = load_mmarray(mmarray_path)

# get the best model among all searched parameter combinations
model_param_dict = pickle.load(open(model_param_dict_path, 'rb'))

searched_params = [json.loads(x) for x in model_param_dict.keys()]
params_test_auc = [(param_combo, param_performance['folds test auc']) for param_combo, (_, param_performance, _, _) in model_param_dict.items()]
params_test_auc.sort(key=lambda x: x[1], reverse=True)
best_param_combo = params_test_auc[0][0]

best_test_model_index = model_param_dict[best_param_combo][0]
best_model = torch.load(os.path.join(results_path, f'{best_test_model_index}.pt'))

# print the best model and its test auc
print(f"Best model parameters: {best_param_combo} with test auc {params_test_auc[0][1]}")
# print the test auc for each combination
for param_combo, test_auc in params_test_auc:
    print(f"{param_combo}: {test_auc}")

# prepare the model to test
criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(best_model, device=device)
if use_ordered:
    test_data = mmarray.get_ordered_test_set(device=device, convert_to_tensor=True)
else:
    test_data = mmarray.get_test_set(device=device, convert_to_tensor=True)
x_eeg_test, y_test = test_data[0], test_data[-1]

is_pca_ica = 'pca' in mmarray['eeg'].data_processor.keys() and 'ica' in mmarray['eeg'].data_processor.keys()
pca = mmarray['eeg'].data_processor['pca'] if is_pca_ica else None
ica = mmarray['eeg'].data_processor['ica'] if is_pca_ica else None

_encoder = mmarray._encoder

montage = mmarray.physio_arrays[0].info['montage']
exg_resample_rate = mmarray['eeg'].sampling_rate
event_names =  list(mmarray.event_viz_colors.keys())
num_channels, num_timesteps = best_model.num_channels, best_model.num_timesteps
eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names

# plot HT viz #########################################################################################################
eeg_viz_picks = ['Fz', 'Cz', 'Oz']

if plot_eeg_epochs:
    visualize_eeg_samples(x_eeg_test, np.array(y_test, dtype=int), mmarray.event_id_viz_colors, this_picks)

if plot_ht_viz:
    # ht_eeg_viz_multimodal(best_model, mmarray, attention_layer_class, device, rollout_data_root,
    #                       note='', load_saved_rollout=False, head_fusion=head_fusion,
    #                       discard_ratio=discard_ratio, is_pca_ica=is_pca_ica, pca=pca, ica=ica, use_meta_info=True)

    ht_eeg_viz_multimodal_batch(best_model, mmarray, attention_layer_class, device, rollout_data_root,
           note='', load_saved_rollout=False, head_fusion='max',
           discard_ratio=0.9,is_pca_ica=is_pca_ica, pca=pca, ica=ica, batch_size=128, use_ordered=use_ordered,
                                eeg_montage=montage, topo_map='forward', roll_topo_map_samples='all', picks=eeg_viz_picks, cls_colors=mmarray.event_viz_colors)

# visualize_eeg_samples(x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], colors, this_picks)
# # a = model(torch.from_numpy(x_eeg_pca_ica_test[viz_indc if viz_both else non_target_indc[0:num_samp]].astype('float32')))
# ht_viz(model, x_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], _encoder, event_names, rollout_data_root, best_model.window_duration,
#        exg_resample_rate,
#        eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max',
#        discard_ratio=0.1, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)


# # plot training history ##########################################################
# for params, history_folds in training_histories.items():
#     params_dict = dict(params)
#     seached_params = [params_dict[key] for key in search_params]
#     for i in range(nfolds):
#         history = {'loss_train': history_folds['loss_train'][i], 'acc_train': history_folds['acc_train'][i], 'loss_val': history_folds['loss_val'][i], 'acc_val': history_folds['acc_val'][i], 'auc_val': history_folds['auc_val'][i], 'auc_test': history_folds['auc_test'][i], 'acc_test': history_folds['acc_test'][i], 'loss_test': history_folds['loss_test'][i]}
#         plot_training_history(history, seached_params, i)
#
#
# # plot ROC curve for each stored model
# if is_plot_ROC:
#     for params, model_list in models.items():
#         for i in range(len(model_list)):
#             model = model_list[i]
#             new_model = HierarchicalTransformer(180, 20, 200, num_classes=2,
#                                     extraction_layers=None,
#                                     depth=params['depth'], num_heads=params['num_heads'],
#                                     feedforward_mlp_dim=params['feedforward_mlp_dim'],
#                                     pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
#                                     dim_head=params['dim_head'], emb_dropout=params['emb_dropout'],
#                                     attn_dropout=params['attn_dropout'], output=params['output'],
#                                     training_mode='classification')
#             model = new_model.load_state_dict(model.state_dict())
#             test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, y_all, y_all_pred = eval(
#                 model, x_eeg_pca_ica_test, y_test, criterion, last_activation, _encoder,
#                 test_name=TaskName.Normal.value, verbose=1)
#             params_dict = dict(params)
#             seached_params = [params_dict[key] for key in search_params]
#             viz_binary_roc(y_all, y_all_pred, seached_params, fold=i)
#
# # plot attention weights
# if is_plot_topomap:
#     rollout_data_root = f'HT_viz'
#     eeg_montage = mne.channels.make_standard_montage('biosemi64')
#     eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
#     num_channels, num_timesteps = x_eeg_pca_ica_test.shape[1:]
#     best_model = models[model_idx[0]][model_idx[1]]
#     ht_viz(best_model, x_eeg_test, y_test, _encoder, event_names, rollout_data_root, best_model.window_duration, exg_resample_rate,
#                    eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=False, head_fusion='max', discard_ratio=0.9, batch_size=64, X_pca_ica=x_eeg_pca_ica_test, pca=pca, ica=ica)
#
#
# # plot performance for different params
# grouped_results = {}
# for key, value in locking_performance.items():
#     params = [dict(key)[x] for x in search_params]
#     grouped_results[tuple(params)] = sum(value[metric]) / len(value[metric])
#
# unique_params = dict([(param_name, np.unique([key[i] for key in grouped_results.keys()])) for i, param_name in enumerate(search_params)])
#
# # Create subplots for each parameter
# fig, axes = plt.subplots(nrows=1, ncols=len(search_params), figsize=(16, 5))
#
# # Plot the bar charts for each parameter
# for i, (param_name, param_values) in enumerate(unique_params.items()):  # iterate over the hyperparameter types
#     axis = axes[i]
#     labels = []
#     auc_values = []
#
#     common_keys = []  # find the intersection of keys for this parameter to avoid biasing the results (needed when the grid search is incomplete)
#     for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
#         other_keys = [list(key) for key, value in grouped_results.items() if key[i] == param_val]
#         [x.remove(param_val) for x in other_keys]
#         other_keys = [tuple(x) for x in other_keys]
#         common_keys.append(other_keys)
#     common_keys = set(common_keys[0]).intersection(*common_keys[1:])
#
#     for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
#         # metric_values = [value for key, value in grouped_results.items() if key[i] == param_val and remove_value(key, param_val) in common_keys]
#         metric_values = []
#         for key, value in grouped_results.items():
#             if key[i] == param_val and tuple(remove_value(key, param_val)) in common_keys:
#                 metric_values.append(value)
#
#         auc_values.append(np.mean(metric_values))
#         labels.append(param_val)
#
#     xticks = np.arange(len(labels))
#     axis.bar(xticks, auc_values)
#     axis.set_xticks(xticks, labels=labels, rotation=45)
#     axis.set_xlabel(param_name)
#     axis.set_ylabel(metric)
#     axis.set_title(f"HT Grid Search")
#
# # Adjust the layout of subplots and show the figure
# fig.tight_layout()
# plt.show()
