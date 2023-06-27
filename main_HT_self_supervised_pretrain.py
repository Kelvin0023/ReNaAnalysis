import datetime
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from renaanalysis.learning.train import eval_model, preprocess_model_data
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples

# test domain

result_path = 'results/model_performances_self_sup'
# analysis parameters ######################################################################################
eeg_resample_rate = 200
reject = 'auto'
bids_root = 'D:/Dataset/auditory_oddball'
event_names = ["standard", "oddball_with_reponse"]
colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}
picks = 'eeg'
# models = ['HT', 'HDCA', 'EEGCNN']
models = ['HT-sesup']
n_folds = 6
ht_lr = 1e-3
ht_l2 = 1e-5

reload_saved_samples = False
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

if reload_saved_samples:
    x, y = get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, eeg_resample_rate, colors)
    with open(os.path.join(f'{export_data_root}','x_auditory_oddball.p'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(f'{export_data_root}','y_auditory_oddball.p'), 'wb') as f:
        pickle.dump(y, f)
else:
    with open(os.path.join(f'{export_data_root}','x_auditory_oddball.p'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(f'{export_data_root}','y_auditory_oddball.p'), 'rb') as f:
        y = pickle.load(f)

event_ids = {'standard': 1, 'oddball_with_reponse': 7}
baseline = (-0.1, 0.0)
ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EOG00', 'EOG01']
eeg_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
info = mne.create_info(ch_names=ch_names, sfreq=256)
# epochs = mne.EpochsArray(data, info)

x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica = preprocess_model_data(x, None)

results = dict()
now = datetime.datetime.now()
datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")
result_path = result_path + datetime_string

for m in models:
    m_results, training_histories = eval_model(x, None, y, event_names, model_name=m, exg_resample_rate=eeg_resample_rate, n_folds=n_folds, ht_lr=ht_lr, ht_l2=ht_l2, eeg_montage=eeg_montage,
                           x_eeg_znormed=x_eeg_znormed, x_eeg_pca_ica=x_eeg_pca_ica, x_pupil_znormed=x_pupil_znormed,
                           test_name=f"auditory_oddball_{m}_{datetime_string}")
    results = {**m_results, **results}
    pickle.dump(results, open(result_path, 'wb'))
    pickle.dump(training_histories, open(result_path + f'{m}_training_history', 'wb'))