# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch

from renaanalysis.learning.grid_search import grid_search_ht_eeg
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
epoch_t_min = -1
epoch_t_max = 20

n_folds = 1
is_pca_ica = False # apply pca and ica on data or not
is_by_channel = False # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_confusion_matrix = False # plot confusion matrix of training and validation during training or not
viz_rebalance = False # viz training data after rebalance or not
is_regenerate_epochs = False # regenerate epochs or not

data_root = 'D:/HaowenWei/Data/HT_Data/fNIRS/FingerFootTapping'

dataset_name = 'fNIRS_finger_foot_tapping'
mmarray_fn = f'{dataset_name}_mmarray_class-weight.p'
task_name = TaskName.TrainClassifier
rebalance_method = None

grid_search_params = {
    "depth": [2],
    "num_heads": [4],
    "pool": ['cls'],
    "feedforward_mlp_dim": [32],

    # "patch_embed_dim": [64, 128, 256],
    "patch_embed_dim": [64],

    "dim_head": [64],
    "attn_dropout": [0.1],
    "emb_dropout": [0.1],
    "lr": [1e-5],
    "l2_weight": [1e-5],

    "pos_embed_mode": ['learnable'],
    # "pos_embed_mode": ['sinusoidal'],

    # "lr_scheduler_type": ['cosine'],
    "lr_scheduler_type": ['cosine'],
    "output": ['multi'],

    "window_duration": [1],
}


# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root, is_regenerate_epochs=is_regenerate_epochs, random_seed=random_seed,
                          rebalance_method=rebalance_method, filename=mmarray_path, epoch_t_min=epoch_t_min, epoch_t_max=epoch_t_max)
    mmarray.save_to_path(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))

locking_performance, training_histories, models = grid_search_ht_eeg(grid_search_params, mmarray, n_folds, task_name=task_name, num_classes=3, physio_type=fnirs_name,
                                                                     is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed)
if task_name == TaskName.PreTrain:
    pickle.dump(training_histories,
                open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(locking_performance,
                open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
else:
    pickle.dump(training_histories, open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(locking_performance, open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))

