# analysis parameters ######################################################################################
import os
import pickle
import time
from datetime import datetime

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch

from renaanalysis.learning.HT import HierarchicalTransformer
from renaanalysis.learning.RHT import RecurrentHierarchicalTransformer
from renaanalysis.learning.grid_search import grid_search_ht_eeg
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters

dataset_params = {
     'dataset_name': 'BCICIVA',
     'data_root': r'D:\Dropbox\Dropbox\EEGDatasets\BCICompetitionIV2a',
     'export_root': export_data_root,
     'reject': None,
     'is_apply_pca_ica_eeg': False,
     'eeg_baseline': False,
     'eeg_resample_rate': 250,
     'random_seed': random_seed,
     'rebalance_method': None,
     'is_regenerate_epochs': True
}


train_params = {
    'n_folds': 1,
    'epochs': 1000,
    'patience': 200,
    'batch_size': 72,
    'use_ordered': False,
    'is_augment_batch': True,
    'use_scheduler': False,
    'test_size': 0.0,
    'val_size': 0.1,  # doesn't matter because we are using predefined splits
    'split_picks': {'subjects': [{'train': ['A03'], 'val': ['A03']}, ], 'run': [{'train': [1], 'val': [2]}, ]},
    'task_name': TaskName.TrainClassifier,
    'model_class': HierarchicalTransformer,
    'is_plot_confusion_matrix': False,

    'encode_y': False,

    'grid_search_params': {
        "depth": [4],
        "num_heads": [8],
        "pool": ['cls'],
        "feedforward_mlp_dim": [256],

        "patch_embed_dim": [64],

        "dim_head": [64],
        "attn_dropout": [0.5],
        "emb_dropout": [0.5],
        "ff_dropout": [0.5],

        "lr": [1e-4],
        "l2_weight": [1e-5],

        "lr_scheduler_type": [None],

        "pos_embed_mode": ['learnable'],
        # "pos_embed_mode": ['sinusoidal'],

        # "output": ['single'],
        "output": ['multi'],

        "time_conv_strid": [0.005],
        'time_conv_window': [0.1],

        "token_recep_field": [0.4],
        "token_recep_field_overlap": [0.3]
    }
}


# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

date_time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
training_results_dir = f"grid_search/{train_params['model_class'].__name__}_{dataset_params['dataset_name']}_{date_time_str}"
training_results_path = os.path.join(os.getcwd(), training_results_dir)
os.makedirs(training_results_path, exist_ok=True)
dataset_params['filename'] = f"{dataset_params['dataset_name']}_mmarray.p"
train_params['training_results_path'] = training_results_path

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, dataset_params['filename'])
if not os.path.exists(mmarray_path):
    train_params['mmarray'] = get_dataset(**dataset_params)
    train_params['mmarray'].save()
else:
    train_params['mmarray'] = pickle.load(open(mmarray_path, 'rb'))


param_performance, training_histories, models = grid_search_ht_eeg(**train_params)
