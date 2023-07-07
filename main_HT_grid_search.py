# analysis parameters ######################################################################################
import copy
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from RenaAnalysis import get_rdf
from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.learning.train import eval_lockings, grid_search_ht
from renaanalysis.params.params import *
from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples


# user parameters
exg_resample_rate = 200
'''
grid_search_params = {
    "depth": [2, 4, 6],
    "num_heads": [4, 8, 12],
    "pool": ['cls', 'mean'],
    "feedforward_mlp_dim": [64, 128, 256],

    # "patch_embed_dim": [64, 128, 256],
    "patch_embed_dim": [128, 256, 512],

    "dim_head": [64, 128, 256],
    "attn_dropout": [0.0, 0.2, 0.2],
    "emb_dropout": [0.0, 0.2, 0.2],
    "lr": [1e-4, 1e-3, 1e-2],
    "l2_weight": [1e-6, 1e-5, 1e-4],

    # "lr_scheduler_type": ['cosine'],
    "lr_scheduler_type": ['cosine', 'exponential'],
    "output": ['single', 'multi'],
}
'''
grid_search_params = {
    "depth": [4],
    "num_heads": [8],
    "pool": ['cls'],
    "feedforward_mlp_dim": [32],

    # "patch_embed_dim": [64, 128, 256],
    "patch_embed_dim": [64],

    "dim_head": [128],
    "attn_dropout": [0.5],
    "emb_dropout": [0.5],
    "lr": [1e-3],
    "l2_weight": [1e-5],

    # "lr_scheduler_type": ['cosine'],
    "lr_scheduler_type": ['cosine'],
    "output": ['multi'],
    'temperature' : [0.1],
    'n_neg': [1],
    'p_t': [0.1],
    'p_c': [0.2],
    'mask_t_span': [2],
    'mask_c_span': [5]
}
data_root = 'D:/Dataset/auditory_oddball'
Dataset_name = 'auditory_oddball'
eeg_resample_rate = 200
reject = 'auto'
event_names = ["standard", "oddball_with_reponse"]
colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}
picks = 'eeg'
searched_params = []
for key, value in grid_search_params.items():
    if len(value) > 1:
        searched_params.append(key)
model_name = 'HT-pca-ica' # HT-sesup, HT, HT-pca-ica
test_name = f'Grid_Search-{searched_params}-{Dataset_name}-{model_name}'
locking_filter = [lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Distractor"],
                  lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Target"]]

# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf(exg_resample_rate=exg_resample_rate, ocular_artifact_mode='proxy')
# rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
# pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")


plt.rcParams.update({'font.size': 22})
n_folds = 3
is_pca_ica = True # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_conf = False # plot confusion matrix of training and validation during training or not
viz_rebalance = False # viz training data after rebalance or not
is_regenerate_epochs = False
reload_saved_samples = True

# locking_name_filters_vs = {
#                         'VS-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
#                                 lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
#                         'VS-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
#                                 lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]],
#                         'VS-I-DT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
#                                 lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]],
#                        'VS-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
#                                  lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]}

locking_name_filters_constrained = {
                        'RSVP-Item-Onset': [lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                            lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Target"]],
}
reload_saved_samples = True
regenerate_epochs = False
if regenerate_epochs:
    x, y, start_time, metadata = get_auditory_oddball_samples(data_root, export_data_root, reload_saved_samples, event_names, picks, reject,
                                        eeg_resample_rate, colors)
    pickle.dump(x, open(os.path.join(export_data_root, f'x_auditory_oddball.p'), 'wb'))
    pickle.dump(y, open(os.path.join(export_data_root, f'y_auditory_oddball.p'), 'wb'))
else:
    try:
        x = pickle.load(open(os.path.join(export_data_root, f'x_auditory_oddball.p'), 'rb'))
        y = pickle.load(open(os.path.join(export_data_root, f'y_auditory_oddball.p'), 'rb'))
    except FileNotFoundError:
        raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}")
x_eeg = z_norm_by_trial(x)
if reload_saved_samples == False:
    x_eeg_pca_ica, pca, ica = compute_pca_ica(x_eeg, num_top_components)
    pickle.dump(x_eeg_pca_ica, open(os.path.join(export_data_root, f'x_pca_ica.p'), 'wb'))
    if is_pca_ica:
        with open(f'{export_data_root}/pca_object.p', 'wb') as f:
            pickle.dump(pca, f)
        with open(f'{export_data_root}/ica_object.p', 'wb') as f:
            pickle.dump(ica, f)
else:
    x_eeg_pca_ica = pickle.load(open(os.path.join(export_data_root, f'x_pca_ica.p'), 'rb'))


locking_performance, training_histories, models = grid_search_ht(grid_search_params, x_eeg, x_eeg_pca_ica, y, event_names, n_folds, test_name=test_name, task_name=TaskName.PreTrain, is_pca_ica=is_pca_ica, is_by_channel=is_by_channel, is_plot_conf=is_plot_conf, regenerate_epochs=is_regenerate_epochs, reload_saved_samples=reload_saved_samples, exg_resample_rate=exg_resample_rate, viz_rebalance=viz_rebalance)
if model_name == 'HT-sesup':
    pickle.dump(training_histories,
                open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(locking_performance,
                open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
else:
    pickle.dump(training_histories, open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))
    pickle.dump(locking_performance, open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))




# import pickle
#
# results = pickle.load(open('model_locking_performances', 'rb'))
# new_results = dict()
# for key, value in results.items():
#     new_key = copy.copy(key)
#     if type(key[1]) is not str:
#         i = str(key[1]).index('(')
#         new_key = (new_key[0], str(new_key[1])[:i])
#     new_results[new_key] = value
# pickle.dump(new_results, open('model_locking_performances', 'wb'))
# exit()
# plt.rcParams["figure.figsize"] = (24, 12)
#
# models = ['HDCA EEG', 'HDCA Pupil', 'HDCA EEG-Pupil', 'EEGPupilCNN', 'EEGCNN']
# constrained_conditions = ['RSVP', 'Carousel']
# conditions_names = ['RSVP', 'Carousel', 'VS']
# constrained_lockings = ['Item-Onset', 'I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']
# lockings = ['I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']
#
# width = 0.175
#
# for c in conditions_names:
#     this_lockings = lockings if c not in constrained_conditions else constrained_lockings
#     ind = np.arange(len(this_lockings))
#
#     for m_index, m in enumerate(models):
#         aucs = [results[(f'{c}-{l}', m)]['folds val auc'] for l in this_lockings]  # get the auc for each locking
#
#         plt.bar(ind + m_index * width, aucs, width, label=f'{m}')
#         for j in range(len(aucs)):
#             plt.text(ind[j] + m_index * width, aucs[j] + 0.05, str(round(aucs[j], 3)), horizontalalignment='center',verticalalignment='center')
#
#     plt.ylim(0.0, 1.1)
#     plt.ylabel('AUC')
#     plt.title(f'Condition {c}, Accuracy by model and lockings')
#     plt.xticks(ind + width / 2, this_lockings)
#     plt.legend(loc=4)
#     plt.show()