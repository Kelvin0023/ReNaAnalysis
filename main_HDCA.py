# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch
from mne.viz import plot_topomap
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from sklearn.model_selection import train_test_split

from renaanalysis.learning.train import prepare_sample_label
from eye.eyetracking import GazeRayIntersect
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
x, y, _ = prepare_sample_label(rdf, event_names, event_filters, picks=None)  # pick all EEG channels
# pickle.dump(x, open('x_p1_s2_flg.p', 'wb'))
# pickle.dump(y, open('y_p1_s2_flg.p', 'wb'))
pickle.dump(x, open('x_constrained.p', 'wb'))
pickle.dump(y, open('x_constrained.p', 'wb'))

# x = pickle.load(open('x_constrained.p', 'rb'))
# y = pickle.load(open('y_constrained.p', 'rb'))

split_window=100e-3

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# split data into 100ms bins
split_size = int(split_window * exg_resample_srate)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_seed)

x_train_windowed = sliding_window_view(x_train, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
x_test_windowed = sliding_window_view(x_test, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
num_trials, num_channels, num_windows, num_timepoints_per_window = x_train_windowed.shape
# compute Fisher's LD for each temporal window
print("Computing windowed LDA per channel, and project per window and trial")
weights_channel_window_time = np.empty(x_train_windowed.shape[1:3] + (split_size,))
windowProjection_channel_window_trial = np.empty((num_trials, num_channels, num_windows))
for i in range(x_train_windowed.shape[1]): # iterate over channels  # TODO this can go faster with multiprocess pool
    for k in range(num_windows):  # iterate over different windows
        this_x = x_train_windowed[:, i, k, :]
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(this_x, y_train)
        _weights = np.squeeze(lda.coef_, axis=0)

        for j in range(num_trials):
            windowProjection_channel_window_trial[j, i, k] = np.dot(_weights, this_x[j])

activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
for class_index in range(2):
    this_x = x_train_windowed[y_train == class_index]
    this_projection = windowProjection_channel_window_trial[y_train == class_index]
    for i in range(num_channels):
        for j in range(num_windows):
            z0 = this_x[:, i, j, :].T
            z1 = this_projection[:, i, j].reshape((-1, 1))
            a = np.matmul(z0, z1) / np.matmul(z1.T, z1).item()
            activation[class_index, i, j] = np.squeeze(np.matmul(z0, z1) / np.matmul(z1.T, z1).item(), axis=1)

info = mne.create_info(
    eeg_channel_names,
    sfreq=exg_resample_srate,
    ch_types=['eeg'] * len(eeg_channel_names))
info.set_montage(eeg_montage)

fig, axs = plt.subplots(2, num_windows - 1, figsize=(22, 10), sharey=True)  # sharing vmax and vmin
for class_index in range(2):
    for i in range(1, num_windows):
        a = np.mean(activation[class_index, :, i, :], axis=1)
        plot_topomap(a, info, axes=axs[class_index, i-1], show=False, res=512)
        axs[class_index, i-1].set_title(f"{int((i-1) * split_window * 1e3)}-{int(i * split_window * 1e3)}ms")
plt.tight_layout()
plt.show()

# fig, axs = plt.subplots(1, num_windows - 1, figsize=(25, 6), sharey=True)  # sharing vmax and vmin
# for i in range(1, num_windows):
#     a = np.mean(activation[1, :, i, :], axis=1)
#     plot_topomap(a, info, ch_type='eeg', axes=axs[i-1], show=False, res=512)
#     axs[i-1].set_title(f"{int((i-1) * split_window * 1e3)}-{int(i * split_window * 1e3)}ms")
# # plt.title(f"Forward Model from Fisher Linear Discriminant Analysis")
# plt.tight_layout()
# plt.show()

# train classifier, use gradient descent to find the cross-window weights
wegiths_window = torch.rand().to(device)
_windowProjection_channel_window_trial = torch.Tensor(windowProjection_channel_window_trial).to(device)

_windowProjection_channel_window_trial[0]