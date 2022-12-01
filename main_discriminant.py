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

"""
Parameters (in the file /params.py):
@param is_regenerate_ica: whether to regenerate ica for the EEG data, if yes, the script calculates the ica components
while processing the EEG data. The generated ica weights will be save to the data path, so when running the script
the next time and if the EEG data is not changed, you can set this to false to skip recalculating ica to save time
@param tmin_pupil, tmax_pupil, tmin_eeg, tmax_eeg: epoch time window for pupil and EEG
@param tmin_pupil_viz, tmax_pupil_viz, tmax_eeg_viz, tmax_eeg_viz: plotting time window for pupil and EEG
@param eventMarker_conditionIndex_dict: dictionary <key, value>=<condition name, event marker channels>. There are four 
conditions RSVP, carousel, VS, and TS. Each condition has four channels/columns of event markers.
@param base_root, data_directory: all data (directories named 0, 1, 2 etc., the numbers are participant IDs) is in the path: 
base_root + data_directory. 
For example, when 
base_root = "C:/Users/Lab-User/Dropbox/ReNa/data/ReNaPilot-2022Spring/"
data_directory = "Subjects"
, then data will be at "C:/Users/Lab-User/Dropbox/ReNa/data/ReNaPilot-2022Spring/Subjects"
@param exg_srate, eyetracking_srate: the sampling rate of the exg (EEG and ECG) device and the eyetracker
@param eeg_picks: the eeg channels to run analysis (standard 64 channel 10-20 system). We only take the midline
electrodes (xz, and xxz) because reorientation is mostly located in the midline. Alternatively, you can set this 
parameter to 
mne.channels.make_standard_montage('biosemi64').ch_names 
to take all the 64 channels
"""

"""
Note on event marker:
Event markers are encoded in integers, this list shows what event does each number represents
1 is distractor, 2 is target, 3 is novelty
4 and 5 are block starts and ends
6 and 7 encodes fixation and saccade onset respectively

6: fixation onset distractor
7: fixation onset target
8: fixation onset novelty
9: fixation onset null
9: saccade onset
"""



# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = get_rdf()
pickle.dump(rdf, open('rdf.p', 'wb'))

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

pickle.dump(x, open('x.p', 'wb'))
pickle.dump(y, open('y.p', 'wb'))
model = EEGCNNNet(in_length=x.shape[-1], num_classes=2)
score_model(x, y, model)

# statistical difference test  #####################################################
# event_names = ["Distractor", "Target"]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
# r_square_test(rdf, event_names, event_filters, title="Constrained Conditions")
#
# event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to detected fixation (Patch-Sim)")
#
# event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]]
# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to detected fixation (I-VT-Head)")
#
# event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn==dtnn_types["Target"]]
# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to detected fixation (I-VT)")


