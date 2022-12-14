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
from sklearn.model_selection import train_test_split

from RenaAnalysis import prepare_sample_label
from eye.eyetracking import GazeRayIntersect
from params import *

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf()
# rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
# pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))  # dump to the SSD c drive
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")
# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})
# colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_names = ["Distractor", "Target"]
# event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
#
# x, y = prepare_sample_label(rdf, event_names, event_filters, picks=None)  # pick all EEG channels
# pickle.dump(x, open('x_p1_s2_flg.p', 'wb'))
# pickle.dump(y, open('y_p1_s2_flg.p', 'wb'))
# pickle.dump(x, open('x_constrained.p', 'wb'))
# pickle.dump(y, open('y_constrained.p', 'wb'))

x = pickle.load(open('x_constrained.p', 'rb'))
y = pickle.load(open('y_constrained.p', 'rb'))

split_window=100e-3

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# split data into 100ms bins
split_size = int(split_window * exg_resample_srate)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_seed)

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
activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
# for class_index in range(2):  # for test set
#     this_x = x_test_windowed[y_test == class_index]
#     for j in range(num_windows):
#         this_x_window = this_x[:, :, j, :].reshape(this_x.shape[0], -1).T
#         z_window = np.array([np.dot(weights_channelWindow[j], this_x[trial_index, :, j, :].reshape(-1)) for trial_index in range(this_x.shape[0])])
#         z_window = z_window.reshape((-1, 1)) # change to a col vector
#         a = (np.matmul(this_x_window, z_window) / np.matmul(z_window.T, z_window).item()).reshape((num_channels, num_timepoints_per_window))
#         activation[class_index, :, j] = a

for class_index in range(2):  # for training set
    this_x = x_train_windowed[y_train == class_index]
    for j in range(num_windows):
        this_x_window = this_x[:, :, j, :].reshape(this_x.shape[0], -1).T
        z_window = np.array([np.dot(weights_channelWindow[j], this_x[trial_index, :, j, :].reshape(-1)) for trial_index in range(this_x.shape[0])])
        z_window = z_window.reshape((-1, 1)) # change to a col vector
        a = (np.matmul(this_x_window, z_window) / np.matmul(z_window.T, z_window).item()).reshape((num_channels, num_timepoints_per_window))
        activation[class_index, :, j] = a

info = mne.create_info(
    eeg_channel_names,
    sfreq=exg_resample_srate,
    ch_types=['eeg'] * len(eeg_channel_names))
info.set_montage(eeg_montage)

fig = plt.figure(figsize=(22, 10), constrained_layout=True)
subfigs = fig.subfigures(2, 1)
# fig, axs = plt.subplots(2, num_windows - 1, figsize=(22, 10), sharey=True)  # sharing vmax and vmin
for class_index, e_name in enumerate(event_names):
    axes = subfigs[class_index].subplots(1, num_windows - 1, sharey=True)
    for i in range(1, num_windows):
        a = np.mean(activation[class_index, :, i, :], axis=1)
        plot_topomap(a, info, axes=axes[i-1], show=False, res=512, vlim=(np.min(activation), np.max(activation)))
        axes[i-1].set_title(f"{int((i-1) * split_window * 1e3)}-{int(i * split_window * 1e3)}ms")
    subfigs[class_index].suptitle(e_name)
fig.suptitle("Activation map from Fisher Discriminant Analysis: Training Set", fontsize='x-large')
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
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out
_y_train = torch.Tensor(np.expand_dims(y_train, axis=1)).to(device)
_projectionTrain_window_trial = torch.Tensor(projectionTrain_window_trial).to(device)
_projectionTest_window_trial = torch.Tensor(projectionTest_window_trial).to(device)
model = linearRegression(num_windows, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = torch.sigmoid(model(_projectionTrain_window_trial))
    loss = criterion(y_pred, _y_train)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch}, loss is {loss.item()}")

y_pred = torch.sigmoid(model(_projectionTest_window_trial))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred.detach().cpu().numpy())
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,  estimator_name='example estimator')
display.plot()
plt.show()

cross_window_weights = model.linear.weight.detach().cpu().numpy()[0, :]
plt.plot(cross_window_weights)
plt.show()