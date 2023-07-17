# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch

from renaanalysis.learning.Conformer import Conformer
from renaanalysis.learning.grid_search import grid_search_ht_eeg, grid_search_eeg
from renaanalysis.multimodal.train_multimodal import train_test_classifier_multimodal, train_test_augmented
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_dataset

# user parameters
n_folds = 3
is_pca_ica = False  # apply pca and ica on data or not
is_by_channel = False  # use by channel version of SMOT rebalance or not, no big difference according to experiment and ERP viz
is_plot_confusion_matrix = False  # plot confusion matrix of training and validation during training or not
viz_rebalance = False  # viz training data after rebalance or not
is_regenerate_epochs = True

eeg_resample_rate = 200

reject = 'auto'  # whether to apply auto rejection
# reject = None  # whether to apply auto rejection
# data_root = r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro'
data_root = 'D:\Dataset\BCICIV_2a'
# data_root = 'J:\TUEH\edf'
dataset_name = 'BCICIV'
mmarray_fn = f'{dataset_name}_mmarray.p'
task_name = TaskName.TrainClassifier
subject_pick = None
subject_group_picks = ['001', '002']

# start of the main block ######################################################
plt.rcParams.update({'font.size': 22})

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path) or is_regenerate_epochs:
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root, reject=reject, eeg_resample_rate=250,
                          is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs,
                          subject_picks=subject_pick, subject_group_picks=subject_group_picks, random_seed=random_seed)
    mmarray.save(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))

# define model class
model = Conformer()

# locking_performance, training_histories, models = grid_search_ht_eeg(grid_search_params, mmarray, n_folds, task_name=task_name,
#                                                                      is_plot_confusion_matrix=is_plot_confusion_matrix, random_seed=random_seed)
assert eeg_name in mmarray.keys(), f"grid_search_ht_eeg: {eeg_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
test_name = 'Conformer-train'

mmarray.train_test_split(test_size=0.1, random_seed=random_seed)
eeg_num_channels, eeg_num_timesteps = mmarray['eeg'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['eeg'].array.shape[1:]
eeg_fs = mmarray['eeg'].sampling_rate

total_training_histories = {}
models_param = {}
locking_performance = {}


models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_augmented(
                                                                                            mmarray, model, test_name, task_name=task_name, n_folds=n_folds,
                                                                                            is_plot_conf_matrix=is_plot_confusion_matrix,
                                                                                             verbose=1, lr=lr, l2_weight=l2_weight, random_seed=random_seed)

if task_name == TaskName.PreTrain:
    pickle.dump(training_histories,
                open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(locking_performance,
                open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain.p', 'wb'))
else:
    pickle.dump(training_histories,
                open(f'HT_grid/model_training_histories_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(locking_performance,
                open(f'HT_grid/model_locking_performances_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))
    pickle.dump(models, open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}.p', 'wb'))

