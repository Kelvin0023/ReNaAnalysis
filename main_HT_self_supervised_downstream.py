import datetime
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from renaanalysis.learning.train import eval_model, cv_train_test_model
from renaanalysis.learning.preprocess import preprocess_samples_eeg_pupil
from renaanalysis.multimodal.train_multimodal import train_test_classifier_multimodal
from renaanalysis.params.params import *
from renaanalysis.utils.data_utils import z_norm_by_trial
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples, get_dataset
from renaanalysis.learning.HT import HierarchicalTransformer

# test domain

result_path = 'results/model_performances_self_sup/downstream'
task_name = 'oddball'
# analysis parameters ######################################################################################
eeg_resample_rate = 200
reject = 'auto'
bids_root = 'D:/Dataset/auditory_oddball'
event_names = ["standard", "oddball_with_response"]
colors = {
    "standard": "red",
    "oddball_with_response": "green"
}
picks = 'eeg'
# models = ['HT', 'HDCA', 'EEGCNN']
models = ['HT-sesup']
model_path = 'renaanalysis/learning/saved_models/oddball-pretrainbendr_lr_0.0001_dimhead_128_feeddim_128_numheads_8_patchdim_128_fold_0_pca_False.pt'
n_folds = 1
lr = 1e-3
ht_l2 = 1e-5

is_regenerate_epochs = False
is_pca_ica = False
is_by_channel = False
is_plot_conf_matrix = False
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

dataset_name = 'auditory_oddball'
mmarray_fn = f'{dataset_name}_mmarray.p'
data_root = 'D:/Dataset/auditory_oddball'

mmarray_path = os.path.join(export_data_root, mmarray_fn)
if not os.path.exists(mmarray_path):
    mmarray = get_dataset('auditory_oddball', epochs_root=export_data_root, dataset_root=data_root, reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs)
    mmarray.save(mmarray_path)
else:
    mmarray = pickle.load(open(mmarray_path, 'rb'))


ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EOG00', 'EOG01']
eeg_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
pretrained_model = torch.load(model_path).HierarchicalTransformer

models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal(mmarray, pretrained_model, test_name='', task_name=TaskName.PretrainedClassifierFineTune,
                                                                                             n_folds=n_folds, lr=lr, is_plot_conf_matrix=is_plot_conf_matrix, random_seed=random_seed)