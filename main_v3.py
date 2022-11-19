

import json
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import mne

from eye.EyeUtils import temporal_filter_fixation
from eye.eyetracking import gaze_event_detection, gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from utils.RenaDataFrame import RenaDataFrame
from utils.data_utils import get_exg_data
from utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from utils.utils import generate_pupil_event_epochs, \
    flatten_list, generate_eeg_event_epochs, visualize_pupil_epochs, visualize_eeg_epochs, \
    read_file_lines_as_list, get_gaze_ray_events, get_item_events, rescale_merge_exg, extract_block_data
from params import *
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events

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

is_loading_saved_analysis = False

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# get the list of paths to save the analysis results
preloaded_dats_path, preloaded_epoch_path, preloaded_block_path, gaze_statistics_path, gaze_behavior_path, epoch_data_export_root = get_analysis_result_paths(base_root, note)
# get the paths to the data files
participant_list, participant_session_file_path_dict, participant_badchannel_dict = get_data_file_paths(base_root, data_directory)


# init variables to hold data
participant_condition_epoch_dict = defaultdict(dict)  # participants -> condition name -> epoch object
participant_condition_block_dict = defaultdict(dict)
condition_gaze_statistics = defaultdict(lambda: defaultdict(list))
condition_gaze_behaviors = defaultdict(lambda: defaultdict(list))


# preload all the .dats or .p

rdf = RenaDataFrame()

if not is_loading_saved_analysis:

    participant_session_file_path_dict = load_participant_session_dict(participant_session_file_path_dict, preloaded_dats_path)
    print("Loading data took {0} seconds".format(time.time() - start_time))

    for p_i, (participant_index, session_dict) in enumerate(participant_session_file_path_dict.items()):
        # print("Working on participant {0} of {1}".format(int(participant_index) + 1, len(participant_session_dict)))
        for session_index, session_files in session_dict.items():
            print("Processing participant-code[{0}]: {4} of {1}, session {2} of {3}".format(int(participant_index),len(participant_session_file_path_dict),session_index + 1,len(session_dict), p_i + 1))
            data, item_catalog_path, session_log_path, session_bad_eeg_channels_path, session_ICA_path = session_files
            session_bad_eeg_channels = open(session_bad_eeg_channels_path, 'r').read().split(' ') if os.path.exists(session_bad_eeg_channels_path) else None
            item_catalog = json.load(open(item_catalog_path))
            session_log = json.load(open(session_log_path))
            item_codes = list(item_catalog.values())

            # markers
            events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])

            # add gaze behaviors from I-DT
            events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events)
            # events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
            # add gaze behaviors from patch sim
            events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)

            # visualize_gaze_events(events, 6)
            rdf.add_participant_session(data, events, participant_index, session_index, session_bad_eeg_channels, session_ICA_path)  # also preprocess the EEG data

# rdf.preprocess()
colors = ['blue', 'red']

event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
# rdf.viz_pupil_epochs(["Distractor", "Target"], event_filters, colors, participant='1', session=1)
rdf.viz_pupil_epochs(["Distractor", "Target"], event_filters, colors, participant=['0', '1'], session=[0, 1])
rdf.viz_eeg_epochs(["Distractor", "Target"], event_filters, colors)



end_time = time.time()
print("Took {0} seconds".format(end_time - start_time))


