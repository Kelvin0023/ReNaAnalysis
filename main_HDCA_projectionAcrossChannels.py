# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mne.viz import plot_topomap
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from RenaAnalysis import prepare_sample_label, compute_forward, plot_forward, solve_crossbin_weights
from eye.eyetracking import GazeRayIntersect
from learning.train import rebalance_classes
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
# event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
#
x, y, group = prepare_sample_label(rdf, event_names, event_filters, picks=None)  # pick all EEG channels
# pickle.dump(x, open('x_p1_s2_flg.p', 'wb'))
# pickle.dump(y, open('y_p1_s2_flg.p', 'wb'))
pickle.dump(x, open('x_constrained.p', 'wb'))
pickle.dump(y, open('y_constrained.p', 'wb'))

x = pickle.load(open('x_constrained.p', 'rb'))
y = pickle.load(open('y_constrained.p', 'rb'))

split_window=100e-3


# split data into 100ms bins
split_size = int(split_window * exg_resample_srate)
folds = 10
# multi-fold validation'
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_seed)

for train, test in sgkf.split(x, y, groups=groups):

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_seed)

x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class
x_train_windowed = sliding_window_view(x_train, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
x_test_windowed = sliding_window_view(x_test, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_train_windowed.shape
num_test_trials = len(x_test)
# compute Fisher's LD for each temporal window
print("Computing windowed LDA per channel, and project per window and trial")
weights_channelWindow = np.empty((num_windows, num_channels * num_timepoints_per_window))
projectionTrain_window_trial = np.empty((num_train_trials, num_windows))
projectionTest_window_trial = np.empty((num_test_trials, num_windows))
 # TODO this can go faster with multiprocess pool
for k in range(num_windows):  # iterate over different windows
    this_x_train = x_train_windowed[:, :, k, :].reshape((num_train_trials, -1))
    this_x_test = x_test_windowed[:, :, k, :].reshape((num_test_trials, -1))
    lda = LinearDiscriminantAnalysis(solver='svd')
    lda.fit(this_x_train, y_train)
    _weights = np.squeeze(lda.coef_, axis=0)
    weights_channelWindow[k] = _weights
    for j in range(num_train_trials):
        projectionTrain_window_trial[j, k] = np.dot(_weights, this_x_train[j])
    for j in range(num_test_trials):
        projectionTest_window_trial[j, k] = np.dot(_weights, this_x_test[j])

print('Computing forward model from window projections for test set')
activation = compute_forward(x_test_windowed, y_test, weights_channelWindow)
plot_forward(activation, event_names, split_window, num_windows)

# train classifier, use gradient descent to find the cross-window weights
cross_window_weights, roc_auc, fpr, tpr = solve_crossbin_weights(projectionTrain_window_trial, projectionTest_window_trial, y_train, y_test, num_windows)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
fig = plt.figure(figsize=(10, 10), constrained_layout=True)
display.plot(ax=plt.gca())
plt.tight_layout()
plt.title("")
plt.show()

plt.plot(cross_window_weights)
plt.xticks(ticks=list(range(1, num_windows + 1)), labels=[str(x) for x in list(range(1, num_windows + 1))])
plt.xlabel("100ms windowed bins")
plt.ylabel("Cross-bin weights")
plt.tight_layout()
plt.show()
