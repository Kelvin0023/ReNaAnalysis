# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch

from renaanalysis.learning.Conformer import Conformer
from renaanalysis.learning.grid_search import grid_search_ht_eeg, grid_search_eeg
from renaanalysis.multimodal.ChannelSpace import create_discretize_channel_space
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
n_folds = 1
is_pca_ica = False # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_confusion_matrix = False # plot confusion matrix of training and validation during training or not
viz_rebalance = False # viz training data after rebalance or not
is_regenerate_epochs = True
is_augment_batch = False

eeg_resample_rate = 200

reject = 'auto'  # whether to apply auto rejection
# reject = None  # whether to apply auto rejection
# data_root = r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro'
# data_root = 'D:/Dataset/auditory_oddball'
# data_root = 'J:\TUEH\edf'
data_root = 'D:\Dataset\BCICIV_2a'
# dataset_name = 'auditory_oddball'
# dataset_name = 'TUH'
dataset_name = 'BCICIVA'
# mmarray_fn = f'{dataset_name}_mmarray_smote_pica.p'
mmarray_fn = f'{dataset_name}_mmarray.p'
# rebalance_method = 'class_weight'
# rebalance_method = 'smote'
rebalance_method = None

task_name = TaskName.PreTrain
task_name = TaskName.TrainClassifier
# subject_pick = ['aaaaaaec', 'aaaaaaed', 'aaaaaaee', 'aaaaaaef', 'aaaaaaeg']
subject_pick = None
# subject_group_picks = None
subject_group_picks = ['001']
picks = {'subjects': [{'train': [3], 'val': [3]}, ], 'run': [{'train': [1], 'val': [2]}, ]}
# picks = None
test_size = 0
val_size = 0.1
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
# grid_search_params = {
#     "depth": [4, 6],
#     "num_heads": [8],
#     "pool": ['cls'],
#     "feedforward_mlp_dim": [32],
#
#     # "patch_embed_dim": [64, 128, 256],
#     "patch_embed_dim": [64],
#
#     "dim_head": [128],
#     "attn_dropout": [0.5],
#     "emb_dropout": [0.5],
#     "lr": [1e-4],
#     "l2_weight": [1e-5],
#
#     # "lr_scheduler_type": ['cosine'],
#     "lr_scheduler_type": ['cosine'],
#     "output": ['multi'],
#     'temperature' : [0.1],
#     'n_neg': [1],
#     'p_t': [0.1],
#     'p_c': [0.25],
#     'mask_t_span': [1],
#     'mask_c_span': [5]
# }

grid_search_params = {
    "depth": [2],
    "num_heads": [8],
    "pool": ['cls'],
    "feedforward_mlp_dim": [32],

    # "patch_embed_dim": [64, 128, 256],
    "patch_embed_dim": [64],

    # "pos_embed_mode": ['learnable'],
    "pos_embed_mode": ['sinusoidal'],

    "dim_head": [128],

    # "attn_dropout": [0.5],
    "attn_dropout": [0.3],

    # "emb_dropout": [0.5],
    "emb_dropout": [0.3],

    "lr": [1e-3],
    # "lr": [1e-3],

    "l2_weight": [1e-5],

    "lr_scheduler_type": ['cosine'],
    "output": ['multi'],
    'temperature' : [0.1],
    'n_neg': [20],
    'p_t': [0.5],
    'p_c': [0.5],
    'mask_t_span': [1],
    'mask_c_span': [1]
}


# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root, reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs, subject_picks=subject_pick, subject_group_picks=subject_group_picks, random_seed=random_seed, filename=mmarray_path, rebalance_method=rebalance_method, consensus=np.linspace(0, 1., 11))
    create_discretize_channel_space(mmarray['eeg'])
    mmarray.save_to_path(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))

# train_indices = []
# val_indices = []
# for i in range(9):
#     for j in range(2):
#         if j == 0:
#             train_indices += mmarray.get_indices_by_subject_run(i+1, j+1)
#         else:
#             val_indices += mmarray.get_indices_by_subject_run(i+1, j+1)

locking_performance, training_histories, models = grid_search_ht_eeg(grid_search_params, mmarray, n_folds, task_name=task_name, is_pca_ica=is_pca_ica, test_size=test_size, val_size=val_size,
                                                                     is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed, picks=picks, is_augment_batch=is_augment_batch)
# locking_performance, training_histories, models = grid_search_eeg(grid_search_params, mmarray, model_class, n_folds, task_name=task_name,
#                                                                      is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed)
if task_name == TaskName.PreTrain:
    pickle.dump(training_histories,
                open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_{dataset_name}.p', 'wb'))
    pickle.dump(locking_performance,
                open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_{dataset_name}.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_HTAE_{dataset_name}.p', 'wb'))
else:
    pickle.dump(training_histories, open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(locking_performance, open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))

