import os
import pickle
import time

import torch

from RenaAnalysis import get_rdf, r_square_test
from RenaAnalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.params.params import *
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################

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


torch.manual_seed(random_seed)
np.random.seed(random_seed)

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf()
rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
# rdf = pickle.dump(os.path.join(export_data_root, 'rdf.p'),, open('rdf.p', 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# discriminant test  ####################################################################################################
event_names = ["Distractor", "Target"]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]

# statistical difference test  #####################################################
event_names = ["Distractor", "Target"]
event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="Constrained conditions (RSVP & Carousel), locked to target onset")
#
event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (Patch-Sim)", participant='1', session=2)

event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (I-VT-Head)", participant='1', session=2)

event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (I-VT)", participant='1', session=2)


event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                     lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to FLGI", participant='1', session=2)
