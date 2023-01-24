# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mne.decoding import UnsupervisedSpatialFilter
from mne.viz import plot_topomap
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

from RenaAnalysis import compute_forward, plot_forward, solve_crossbin_weights, \
    compute_window_projections, get_rdf
from utils.data_utils import compute_pca_ica
from eye.eyetracking import GazeRayIntersect, Fixation
from learning.train import rebalance_classes, prepare_sample_label
from params import *

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf()
rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
# pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))  # dump to the SSD c drive
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")
# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})
# colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_names = ["Distractor", "Target"]
event_filters =[lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                     lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]

x, y = prepare_sample_label(rdf, event_names, event_filters, picks=None, participant='1', session=2, data_type='both')  # pick all EEG channels
# x, y, groups = prepare_sample_label(rdf, event_names, event_filters, picks=None)  # pick all EEG channels
pickle.dump(x, open('x_p1_s2_FLGI.p', 'wb'))
pickle.dump(y, open('y_p1_s2_FLGI.p', 'wb'))
# pickle.dump(groups, open('g_p1_s2_FLGI.p', 'wb'))

# pickle.dump(x, open('x_allParticipantSessions_constrained_ItemLocked.p', 'wb'))
# pickle.dump(y, open('y_allParticipantSessions_constrained_ItemLocked.p', 'wb'))
# pickle.dump(groups, open('g_allParticipantSessions_constrained_ItemLocked.p', 'wb'))

# x = pickle.load(open('x_allParticipantSessions_constrained_ItemLocked.p', 'rb'))
# y = pickle.load(open('y_allParticipantSessions_constrained_ItemLocked.p', 'rb'))
# groups = pickle.load(open('g_allParticipantSessions_constrained_ItemLocked.p', 'rb'))

# split data into 100ms bins
split_size = int(split_window * exg_resample_srate)
# multi-fold cross-validation
cross_val_folds = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
_, num_eeg_channels, num_windows, num_timepoints_per_window = sliding_window_view(x, window_shape=split_size, axis=2)[:, :, 0::split_size, :].shape

cw_weights_folds = np.empty((num_folds, num_windows))
activations_folds = np.empty((num_folds, len(event_names), num_eeg_channels, num_windows, num_timepoints_per_window))
roc_auc_folds = np.empty(num_folds)
fpr_folds = []
tpr_folds = []

x_eeg_transformed = compute_pca_ica(x[0], num_top_compoenents)  # apply ICA and PCA

for i, (train, test) in enumerate(cross_val_folds.split(x, y, groups=groups)):  # cross-validation; group arguement is not necessary unless using grouped folds
    print(f"Working on {i+1} fold of {num_folds}")

    x_eeg_transformed_train, x_eeg_transformed_test, y_train, y_test = x_eeg_transformed[train], x_eeg_transformed[test], y[train], y[test]
    x_pupil_train, x_pupil_test, y_train, y_test = x[1][train], x[1][test], y[train], y[test]

    x_eeg_transformed_train, y_train = rebalance_classes(x_eeg_transformed_train, y_train)  # rebalance by class
    x_pupil_train, y_train = rebalance_classes(x_pupil_train, y_train)  # rebalance by class

    x_eeg_test = x[0][test]

    x_eeg_transformed_train_windowed = sliding_window_view(x_eeg_transformed_train, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
    x_eeg_transformed_test_windowed = sliding_window_view(x_eeg_transformed_test, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
    x_eeg_test_windowed = sliding_window_view(x_eeg_test, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window

    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_eeg_transformed_train_windowed.shape
    num_test_trials = len(x_eeg_transformed_test)
    # compute Fisher's LD for each temporal window
    print("Computing windowed LDA per channel, and project per window and trial")
    weights_channelWindow, projectionTrain_window_trial, projectionTest_window_trial = compute_window_projections(x_eeg_transformed_train_windowed, x_eeg_transformed_test_windowed, y_train)
    print('Computing forward model from window projections for test set')
    activation = compute_forward(x_eeg_test_windowed, y_test, projectionTest_window_trial)
    # train classifier, use gradient descent to find the cross-window weights
    # z-norm the projections
    projection_mean = np.mean(np.concatenate([projectionTrain_window_trial, projectionTest_window_trial], axis=0), axis=0, keepdims=True)
    projection_std = np.std(np.concatenate([projectionTrain_window_trial, projectionTest_window_trial], axis=0), axis=0, keepdims=True)

    projectionTrain_window_trial = (projectionTrain_window_trial - projection_mean) / projection_std
    projectionTest_window_trial = (projectionTest_window_trial - projection_mean) / projection_std

    print('Solving cross bin weights')
    cw_weights, roc_auc, fpr, tpr = solve_crossbin_weights(projectionTrain_window_trial, projectionTest_window_trial, y_train, y_test, num_windows)

    cw_weights_folds[i] = cw_weights
    activations_folds[i] = activation
    roc_auc_folds[i] = roc_auc
    fpr_folds.append(fpr)
    tpr_folds.append(tpr)
    print(f'Fold {i}, auc is {roc_auc}')

plot_forward(np.mean(activations_folds, axis=0), event_names, split_window, num_windows, notes=f"Average over {num_folds}-fold's test set")

print(f"Best cross ROC-AUC is {np.max(roc_auc_folds)}")
best_fold_i = np.argmax(roc_auc_folds)
display = metrics.RocCurveDisplay(fpr=fpr_folds[best_fold_i], tpr=tpr_folds[best_fold_i], roc_auc=roc_auc_folds[best_fold_i], estimator_name='example estimator')
fig = plt.figure(figsize=(10, 10), constrained_layout=True)
display.plot(ax=plt.gca(), name='ROC')
plt.tight_layout()
plt.title("ROC of the best cross-val fold")
plt.show()

fig = plt.figure(figsize=(15, 10), constrained_layout=True)
plt.boxplot(cw_weights_folds)
# plt.plot(cross_window_weights)
x_labels = [f"{int((i - 1) * split_window * 1e3)}ms" for i in range(num_windows)]
x_ticks = np.arange(0.5, num_windows+0.5, 1)
plt.plot(list(range(1, num_windows+1)), np.mean(cw_weights_folds, axis=0), label="folds average")

plt.xticks(ticks=x_ticks, labels=x_labels)
plt.xlabel("100ms windowed bins")
plt.ylabel("Cross-bin weights")
plt.title(f'Cross-bin weights, {num_folds}-fold cross validation')
plt.legend()
plt.tight_layout()
plt.show()
