import os
import pickle
import time
import warnings

import torch
from matplotlib import pyplot as plt

from renaanalysis.learning.HT import HierarchicalTransformer
from renaanalysis.learning.HT_cam_viz import ht_eeg_viz_cam
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.learning.RHT import RelAttention
from renaanalysis.params.params import *
from renaanalysis.utils.utils import remove_value
from renaanalysis.utils.viz_utils import viz_binary_roc, visualize_eeg_samples, \
    plot_training_history_folds, grid_search_bars

metric = 'folds_val_auc'
is_by_channel = False
is_plot_train_history = True
is_plot_ROC = True
is_plot_topomap = True

# params for plotting epoch
is_plot_epochs = True
epoch_plot_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']

num_samp = 2
viz_all_epoch = True
viz_both = False
rollout_data_root = f'HT_viz'

viz_pca_ica = False
results_dir = 'grid_search/RecurrentHierarchicalTransformer_auditory_oddball_10-02-2023-14-12-40'

colors = {1: 'red', 7: 'blue'}

########################################################################################################################
device ='cuda:0' if torch.cuda.is_available() else 'cpu'
mmarray = pickle.load(open(os.path.join(results_dir, 'mmarray.p'), 'rb'))
dataset_name = mmarray.dataset_name

training_histories = pickle.load(open(os.path.join(results_dir, 'training_histories.p'), 'rb'))
model_performance = pickle.load(open(os.path.join(results_dir, 'model_performances.p'), 'rb'))
models = pickle.load(open(os.path.join(results_dir, 'models_with_params.p'), 'rb'))
train_params = pickle.load(open(os.path.join(results_dir, 'train_params.p'), 'rb'))

n_folds = list(models.values())[0].__len__()

criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(list(models.values())[0][0], device='cuda:0' if torch.cuda.is_available() else 'cpu')
test_data = mmarray.get_test_set(device=device, convert_to_tensor=False, **train_params)

# TODO clean up this part about pca ica ################################################################################
x_eeg_test, y_test = test_data['eeg'], test_data['y']
x_test_original = mmarray['eeg'].array[mmarray.get_ordered_test_indices() if train_params['use_ordered'] else mmarray.test_indices]

is_pca_ica = 'pca' in mmarray['eeg'].data_processor.keys() or 'ica' in mmarray['eeg'].data_processor.keys()
if is_pca_ica != viz_pca_ica:
    warnings.warn('The mmarray is not consistent with the viz_pca_ica parameter.')
pca = mmarray['eeg'].data_processor['pca'] if 'pca' in mmarray['eeg'].data_processor.keys() else None
ica = mmarray['eeg'].data_processor['ica'] if 'ica' in mmarray['eeg'].data_processor.keys() else None
_encoder = mmarray._encoder

head_fusion = 'mean'
channel_fusion = 'sum'  # TODO when plotting the cross window activation
sample_fusion = 'sum'  # TODO
discard_ratio = 0.9
########################################################################################################################

print('\n'.join([f"{str(x)}, {y[metric]}" for x, y in model_performance.items()]))


# plot training history
if is_plot_train_history:
    plot_training_history_folds(training_histories)

grid_search_bars(model_performance, train_params['grid_search_params'], metric)


# find the model with best test auc
best_auc = 0
for params, perf in model_performance.items():
    for i in range(train_params['n_folds']):
        if perf['folds test auc'][i] > best_auc:
            model_idx = [params, i]
            best_auc = perf['folds test auc'][i]

# plot epochs and GradCam
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
    model = best_model
    if viz_all_epoch:
        visualize_eeg_samples(x_eeg_test, y_test, colors, this_picks)
        t_start = time.perf_counter()
        ht_eeg_viz_cam(best_model, mmarray, RelAttention, device, rollout_data_root,
                              note='', load_saved_rollout=False, head_fusion=head_fusion, cls_colors=mmarray.event_viz_colors,
                              discard_ratio=discard_ratio, is_pca_ica=is_pca_ica, pca=pca, ica=ica, **train_params)
        print("ht viz batched took {} seconds".format(time.perf_counter() - t_start))
    else:
        visualize_eeg_samples(x_eeg_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], colors, this_picks)
        ht_eeg_viz_cam(model, x_eeg_test[viz_indc if viz_both else non_target_indc], y_test[viz_indc if viz_both else non_target_indc], _encoder, event_names, rollout_data_root, best_model.window_duration,
               exg_resample_rate,
               eeg_montage, num_timesteps, num_channels, note='', load_saved_rollout=True, head_fusion='max',
               discard_ratio=0.1, batch_size=64, is_pca_ica=is_pca_ica, pca=pca, ica=ica)




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

