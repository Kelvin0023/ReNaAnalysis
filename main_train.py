import os
import pickle
import time
from collections import defaultdict

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from RenaAnalysis import get_rdf, r_square_test
from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, Fixation
from learning.models import EEGCNNNet
from learning.train import score_model
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from utils.utils import get_item_events, viz_pupil_epochs, viz_eeg_epochs, epochs_to_class_samples
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event

start_time = time.time()  # record the start time of the analysis

rdf = pickle.load(open('rdf.p', 'rb'))

# discriminant test  ####################################################################################################
event_names = ["Distractor", "Target"]
event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
eeg_epochs, eeg_event_ids = rdf.get_eeg_epochs(event_names, event_filters)

x, y = epochs_to_class_samples(eeg_epochs, eeg_event_ids)
epoch_shape = x.shape[1:]
x = np.reshape(x, newshape=(len(x), -1))
sm = SMOTE(random_state=42)
x, y = sm.fit_resample(x, y)
x = np.reshape(x, newshape=(len(x), ) + epoch_shape)  # reshape back x after resampling

# z-norm along channels
x = (x - np.mean(x, axis=(0, 2), keepdims=True)) / np.std(x, axis=(0, 2), keepdims=True)

# sanity check the channels
x_distractors = x[:, eeg_montage.ch_names.index('CPz'), :][y==0]
x_targets = x[:, eeg_montage.ch_names.index('CPz'), :][y==1]
x_distractors = np.mean(x_distractors, axis=0)
x_targets = np.mean(x_targets, axis=0)
plt.plot(x_distractors)
plt.plot(x_targets)
plt.show()

model = EEGCNNNet(in_length=x.shape[-1], num_classes=2)
score_model(x, y, model)
