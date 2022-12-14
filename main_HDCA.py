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
from RenaAnalysis import prepare_sample_label
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
event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
#
x, y = prepare_sample_label(rdf, event_names, event_filters, picks=None, participant='1', session=2)  # pick all EEG channels
pickle.dump(x, open('x_p1_s2_flg.p', 'wb'))
pickle.dump(y, open('y_p1_s2_flg.p', 'wb'))

# x = pickle.load(open('x.p', 'rb'))
# y = pickle.load(open('y.p', 'rb'))

split_window=100e-3
# split data into 100ms bins
split_size = int(split_window * exg_resample_srate)
x_windowed =sliding_window_view(x, window_shape=split_size, axis=2)[:, :, 0::split_size, :]  # shape = #trials, #channels, #windows, #time points per window
num_windows = x_windowed.shape[2]
num_trials = x.shape[0]
num_channels = x.shape[1]
num_timepoints_per_window = x_windowed.shape[-1]
# compute Fisher's LD for each temporal window
print("Computing windowed LDA per channel, and project per window and trial")
weights_channel_window_time = np.empty(x_windowed.shape[1:3]+ (split_size,))
projections_channel_window_trial = np.empty((x.shape[0], num_channels, num_windows))
for i in range(x_windowed.shape[1]): # iterate over channels  # TODO this can go faster with multiprocess pool
    for k in range(num_windows):  # iterate over different windows
        this_x = x_windowed[:, i, k, :]
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(this_x, y)
        _weights = np.squeeze(lda.coef_, axis=0)

        for j in range(num_trials):
            projections_channel_window_trial[j, i, k] = np.dot(_weights, this_x[j])

activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
for class_index in range(2):
    this_x = x_windowed[y==class_index]
    this_projection = projections_channel_window_trial[y==class_index]
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

# train classifier
