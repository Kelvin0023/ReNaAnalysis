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
from renaanalysis.learning.models import EEGInceptionNet
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
is_regenerate_epochs = True
task_name = TaskName.TrainClassifier

model_class_params = [
    ['EEGNet', EEGInceptionNet,
     {}
     ],

    ['Vanilla HT', HierarchicalTransformer,
        {
        "depth": 4,
        "num_heads": 8,
        "pool": 'cls',
        "feedforward_mlp_dim": 256,
        "patch_embed_dim": 256,
        "dim_head": 128,
        "attn_dropout": 0.0,
        "emb_dropout": 0.5,
        "ff_dropout": 0.1,
        "lr": 1e-4,
        "l2_weight": 1e-5,
        "lr_scheduler_type": 'cosine',
        "pos_embed_mode": 'learnable',
        "output": 'multi',
        "time_conv_strid": 0.005,
        "token_recep_field": 0.4}
     ],

    ['RHT w/o mem', RecurrentHierarchicalTransformer,
        {
            "depth": 4,
            "num_heads": 8,
            "pool": 'cls',
            "feedforward_mlp_dim": 256,
            "patch_embed_dim": 256,
            "dim_head": 128,
            "attn_dropout": 0.0,
            "emb_dropout": 0.5,
            "ff_dropout": 0.1,
            "lr": 1e-4,
            "l2_weight": 1e-5,
            "lr_scheduler_type": 'cosine',
            "pos_embed_mode": 'sinusoidal',  # RHT is using sinusoidal positional embedding
            "output": 'multi',}
    ],

    ['RHT w/ mem', RecurrentHierarchicalTransformer,
        {
            "depth": 4,
            "num_heads": 8,
            "pool": 'cls',
            "feedforward_mlp_dim": 256,
            "patch_embed_dim": 256,
            "dim_head": 128,
            "attn_dropout": 0.0,
            "emb_dropout": 0.5,
            "ff_dropout": 0.1,
            "lr": 1e-4,
            "l2_weight": 1e-5,
            "lr_scheduler_type": 'cosine',
            "pos_embed_mode": 'sinusoidal',  # RHT is using sinusoidal positional embedding
            "output": 'multi',
            "mem_len": 1}
    ]
]

train_params = {
    'epochs': 1000,
    'patience': 200,
    'batch_size': 16,
    'use_ordered_batch': False,
    'is_augment_batch': False,
    'use_scheduler': False,
    'test_size': 0.1,
    'val_size': 0.1,
    'n_folds': 3,
}

datasets = {
    'Motor Imagery':
        {'dataset_name': 'BCICIVA',
         'data_root': r'D:\Dropbox\Dropbox\EEGDatasets\BCICompetitionIV2a',
         'export_root': export_data_root,
         'reject': None,
         'is_apply_pca_ica_eeg': False,
         'eeg_baseline': False,
         'eeg_resample_rate': 250,
         'random_seed': random_seed,
         'rebalance_method': None,
         'is_regenerate_epochs': is_regenerate_epochs},
    'Auditory Oddball':
        {'dataset_name': 'auditory_oddball',
         'data_root': r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro',
         'export_root': export_data_root,
         'reject': 'auto',
         'is_apply_pca_ica_eeg': False,
         'eeg_resample_rate': 200,
         'random_seed': random_seed,
         'rebalance_method': None,
         'is_regenerate_epochs': is_regenerate_epochs},
}

is_pca_ica = False # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_regenerate_epochs = True

eeg_baseline = None

eeg_resample_rate = 250

reject = None  # whether to apply auto rejection


date_time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
training_results_dir = f'model_comparison/{date_time_str}'
training_results_path = os.path.join(os.getcwd(), training_results_dir)
os.makedirs(training_results_path, exist_ok=True)


# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root,
                          reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs,
                          random_seed=random_seed, rebalance_method=None, filename=mmarray_path,
                          eeg_baseline=eeg_baseline)
    mmarray.save()
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))


param_performance, training_histories, models = grid_search_ht_eeg(grid_search_params, mmarray, n_folds,
                                                                    training_results_path=training_results_path,
                                                                   task_name=task_name,
                                                                    is_plot_confusion_matrix=is_plot_confusion_matrix,
                                                                   random_seed=random_seed,
                                                                   model_class=model_class, **train_params)
