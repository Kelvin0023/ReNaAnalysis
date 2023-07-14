# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch

from renaanalysis.learning.grid_search import grid_search_ht_eeg, grid_search_rht_eeg
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
n_folds = 3
is_pca_ica = False # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_confusion_matrix = False # plot confusion matrix of training and validation during training or not
viz_rebalance = False # viz training data after rebalance or not
is_regenerate_epochs = False

eeg_resample_rate = 200

# reject = 'auto'  # whether to apply auto rejection
reject = None  # whether to apply auto rejection
# data_root = 'D:/Dataset/auditory_oddball'
data_root = r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro'
dataset_name = 'auditory_oddball'
mmarray_fn = f'{dataset_name}_mmarray_class-weight.p'

grid_search_params = {
    "depth": [4],
    "num_heads": [8],
    "pool": ['cls'],
    "feedforward_mlp_dim": [32],

    # "patch_embed_dim": [64, 128, 256],
    "patch_embed_dim": [128],

    "dim_head": [64],
    "attn_dropout": [0.5],
    "emb_dropout": [0.5],
    "lr": [1e-4],
    "l2_weight": [1e-5],

    # "lr_scheduler_type": ['cosine'],
    "lr_scheduler_type": ['cosine'],
    "output": ['multi'],
    'temperature' : [0.1],
    'n_neg': [1],
    'p_t': [0.1],
    'p_c': [0.25],
    'mask_t_span': [1],
    'mask_c_span': [5]
}


model_name = 'HT-pca-ica' # HT-sesup, HT, HT-pca-ica

# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset('auditory_oddball', epochs_root=export_data_root, data_root=data_root, reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs, random_seed=random_seed, rebalance_method="class_weight")
    mmarray.save(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))

locking_performance, training_histories, models = grid_search_rht_eeg(grid_search_params, mmarray, n_folds, task_name=TaskName.TrainClassifier,
                                                                     is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed)
# if model_name == 'HT-sesup':
#     pickle.dump(training_histories,
#                 open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
#     pickle.dump(locking_performance,
#                 open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
#     pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
# else:
#     pickle.dump(training_histories, open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))
#     pickle.dump(locking_performance, open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))
#     pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_numhead.p', 'wb'))

