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
from renaanalysis.params.params import *
from renaanalysis.utils.data_utils import z_norm_by_trial
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples
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
ht_lr = 1e-3
ht_l2 = 1e-5

reload_saved_samples = False
is_pca_ica = False
is_by_channel = False
is_plot_conf_matrix = False
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

if reload_saved_samples:
    x, y = get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, eeg_resample_rate, colors)
    x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica = preprocess_samples_eeg_pupil(x, None)
    pickle.dump(x_eeg_pca_ica, open(os.path.join(export_data_root, f'x_pca_ica.p'), 'wb'))
    if is_pca_ica:
        with open(f'{export_data_root}/pca_object.p', 'wb') as f:
            pickle.dump(pca, f)
        with open(f'{export_data_root}/ica_object.p', 'wb') as f:
            pickle.dump(ica, f)
    with open(os.path.join(f'{export_data_root}','x_auditory_oddball.p'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(f'{export_data_root}','y_auditory_oddball.p'), 'wb') as f:
        pickle.dump(y, f)
else:
    with open(os.path.join(f'{export_data_root}','x_auditory_oddball.p'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(f'{export_data_root}','y_auditory_oddball.p'), 'rb') as f:
        y = pickle.load(f)
    x_eeg_znormed = z_norm_by_trial(x)


event_ids = {'standard': 1, 'oddball_with_response': 7}
baseline = (-0.1, 0.0)
ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EOG00', 'EOG01']
eeg_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
info = mne.create_info(ch_names=ch_names, sfreq=256)
# epochs = mne.EpochsArray(data, info)

# prepare test dataset
skf = StratifiedShuffleSplit(n_splits=1, random_state=random_seed)
train, test = [(train, test) for train, test in skf.split(x_eeg_znormed, y)][0]
x_eeg_train = x_eeg_znormed[train] if not is_pca_ica else x_eeg_pca_ica[train]
x_eeg_test = x_eeg_znormed[test] if not is_pca_ica else x_eeg_pca_ica[test]
y_train, y_test = y[train], y[test]
assert np.all(np.unique(y_test) == np.unique(y_train)), "train and test labels are not the same"
assert len(np.unique(y_test)) == len(event_names), "number of unique labels"

#load model
# model_dict = torch.load(model_path)
# pretrained_model = HierarchicalTransformer(180, 64, 200, num_classes=2, extraction_layers=None,
#                                         depth=4, num_heads=8, feedforward_mlp_dim=128,
#                                         pool='cls', patch_embed_dim=128,
#                                         dim_head=128, emb_dropout=0.5, attn_dropout=0.3, output='multi', training_mode='self-sup pretrain')
# pretrained_model.load_state_dict(model_dict)
# pretrained_model.training_mode = 'classification'
pretrained_model = torch.load(model_path)
pretrained_model.training_mode = 'classification'

models, training_histories_folds, criterion, last_activation, _encoder = cv_train_test_model(x_eeg_train, y_train, pretrained_model, n_folds=n_folds, lr=ht_lr, l2_weight=ht_l2, test_name=TaskName.PretrainedClassifierFineTune.value,
                                                                                             X_test=x_eeg_test, Y_test=y_test, is_by_channel=is_by_channel, is_plot_conf_matrix=is_plot_conf_matrix)