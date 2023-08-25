import datetime
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
import torch.nn.functional as F

from renaanalysis.learning.train import eval_model, cv_train_test_model
from renaanalysis.learning.preprocess import preprocess_samples_eeg_pupil
from renaanalysis.multimodal.ChannelSpace import create_discretize_channel_space
from renaanalysis.multimodal.train_multimodal import train_test_classifier_multimodal
from renaanalysis.params.params import *
from renaanalysis.utils.data_utils import z_norm_by_trial
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples, get_dataset

# environment parameters ######################################################################################
dataset_name = 'auditory_oddball'
event_names = ["standard", "oddball_with_response"]
colors = {
    "standard": "red",
    "oddball_with_response": "green"
}
picks = 'eeg'
subject_pick = None
subject_group_picks = None
eeg_resample_rate = 200
models = ['HT-sesup']
is_regenerate_epochs = False

# path parameters ######################################################################################
result_path = 'results/downstream'
task_name = 'oddball'
data_root = 'D:/Dataset/auditory_oddball'
reject = 'auto'
model_path = 'renaanalysis/learning/saved_models/oddball-pretrainbendr_lr_0.0001_dimhead_128_feeddim_128_numheads_8_patchdim_128_fold_0_pca_False.pt'
mmarray_fn = f'{dataset_name}_mmarray_class_weight.p'


# training parameters ######################################################################################
rebalance_method = 'class_weight'
n_folds = 3
lr = 1e-2
ht_l2 = 1e-5
window_duration = 0.1
is_pca_ica = False
is_by_channel = False
is_plot_conf_matrix = False

# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)
pretrained_models = pickle.load(open(f'HT_grid/models_with_params_pca_{is_pca_ica}_chan_{is_by_channel}_pretrain_HT_TUH.p', 'rb'))

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root, reject=reject,
                          is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs,
                          subject_picks=subject_pick, subject_group_picks=subject_group_picks, random_seed=random_seed,
                          filename=mmarray_path, rebalance_method=rebalance_method)
    create_discretize_channel_space(mmarray['eeg'])
    mmarray.save_to_path(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))


for pretrained_model_list in pretrained_models.values():
    for pretrained_model in pretrained_model_list:
        model = pretrained_model.HierarchicalTransformer
        model.adjust_model(mmarray['eeg'].array.shape[-1], mmarray['eeg'].array.shape[1], mmarray['eeg'].sampling_rate, window_duration, 2, 'multi')
        models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal(mmarray, model, test_name='', task_name=TaskName.PretrainedClassifierFineTune,
                                                                                             n_folds=n_folds, lr=lr, is_plot_conf_matrix=is_plot_conf_matrix, random_seed=random_seed, l2_weight=l2_weight)
        pickle.dump(models, open(f'{result_path}/models_auditory_oddball.p', 'wb'))