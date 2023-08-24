# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from renaanalysis.learning.Conformer import Conformer
from renaanalysis.learning.grid_search import grid_search_ht_eeg, grid_search_eeg, grid_search_rht_eeg
from renaanalysis.multimodal.ChannelSpace import create_discretize_channel_space
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
n_folds = 2
is_pca_ica = False # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_confusion_matrix = False # plot confusion matrix of training and validation during training or not
viz_rebalance = False # viz training data after rebalance or not
is_regenerate_epochs = True

eeg_resample_rate = 200

# reject = None  # whether to apply auto rejection
reject = 'auto'  # whether to apply auto rejection
# data_root = r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro'
# data_root = 'D:/Dataset/auditory_oddball'
# data_root = 'J:\TUEH\edf'
# data_root = 'D:\Dataset\BCICIV_2a'
# dataset_name = 'auditory_oddball'
data_root = r'D:\Dropbox\Dropbox\EEGDatasets\Linbi_64chan_EEG_dataset_for-Hengda'
dataset_name = 'SIM'
# dataset_name = 'BCICIVA'
# mmarray_fn = f'{dataset_name}_mmarray_smote_pica.p'

mmarray_fn = f'{dataset_name}_mmarray_class_weight.p'
rebalance_method = 'class_weight'

# task_name = TaskName.PreTrain
task_name = TaskName.TrainClassifier
training_results_dir = 'RHT_grid_search_simEEGfMRIPupil'
training_results_path = os.path.join(os.getcwd(), training_results_dir)
if not os.path.exists(training_results_path):
    os.mkdir(training_results_path)

grid_search_params = {
    # "depth": [6],
    "depth": [4],
    "num_heads": [8],
    "pool": ['cls'],
    "feedforward_mlp_dim": [256],
    # "feedforward_mlp_dim": [32],

    "patch_embed_dim": [128],

    # "dim_head": [64],
    "dim_head": [128],
    "attn_dropout": [0.0],
    "emb_dropout": [0.1],
    "dropout": [0.1],

    "lr": [1e-4],
    # "lr": [1e-3],
    "l2_weight": [1e-5],

    # "lr_scheduler_type": ['cosine'],
    "lr_scheduler_type": ['cosine'],

    # "output": ['single'],
    "output": ['multi'],

    'temperature' : [0.1],
    'n_neg': [1],
    'p_t': [0.1],
    'p_c': [0.25],
    'mask_t_span': [1],
    'mask_c_span': [5],
}

# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root, reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs, random_seed=random_seed, filename=mmarray_path, rebalance_method=rebalance_method, consensus=np.linspace(0., 5., 11))
    create_discretize_channel_space(mmarray['eeg'])
    mmarray.save_to_path(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))


param_performance, training_histories, models = grid_search_rht_eeg(grid_search_params, mmarray, n_folds, training_results_path, task_name=TaskName.TrainClassifier,
                                                                     is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed, batch_size=8)

pickle.dump(training_histories, open(os.path.join(training_results_path, f'model_training_histories_pcaica_{is_pca_ica}_chan_{is_by_channel}.p'), 'wb'))
pickle.dump(param_performance, open(os.path.join(training_results_path, f'model_performances_pcaica_{is_pca_ica}_chan_{is_by_channel}.p'), 'wb'))

