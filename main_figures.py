import os
import time
from collections import defaultdict

from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, Fixation, GazeRayIntersect
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from utils.utils import get_item_events, viz_pupil_epochs, viz_eeg_epochs, get_rdf
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event
import matplotlib.pyplot as plt

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

colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Novelty"]]
# viz_eeg_epochs(rdf, ["Distractor", "Target", "Novelty"], event_filters, colors, participant='1', session=1)
viz_eeg_epochs(rdf, ["Distractor", "Target", "Novelty"], event_filters, colors)
#
#
# visualize_rdf_gaze_event(rdf, participant='1', session=1, block_id=6)
#
# event_filters = [lambda x: type(x)==Fixation and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==Fixation and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"],
#                  lambda x: type(x)==Fixation and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Novelty"]]
# viz_pupil_epochs(rdf, ["Distractor", "Target", "Novelty"], event_filters, colors)
#
#
# event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
# viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors)

event_filters = [lambda x: type(x) == Fixation and x.block_condition == conditions[
    'VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"],
                 lambda x: type(x) == Fixation and (
                             x.block_condition == conditions['RSVP'] or x.block_condition == conditions[
                         'Carousel']) and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]
viz_eeg_epochs(rdf, ["VS-Target", "OtherTarget"], event_filters,
               {'VS-Target': 'blue', 'OtherTarget': 'red', 'Novelty': 'orange'},
               title='Locked to Detected Fixation (using patch-similarity)')

event_filters = [
    lambda x: type(x) == Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn ==
              dtnn_types["Distractor"],
    lambda x: type(x) == Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn ==
              dtnn_types["Target"]]
viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='Locked to Detected Fixation (using I-DT)')

event_filters = [
    lambda x: type(x) == Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn ==
              dtnn_types["Target"],
    lambda x: type(x) == Fixation and x.block_condition == conditions[
        'VS'] and x.detection_alg == 'I-DT-Head' and x.dtn == dtnn_types["Target"],
    lambda x: type(x) == Fixation and x.block_condition == conditions[
        'VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]
viz_eeg_epochs(rdf, ["I-DT", "I-DT-Head", "Patch-Sim"], event_filters,
               {'I-DT': 'blue', 'I-DT-Head': 'red', 'Patch-Sim': 'orange'},
               title='Visual Search with Different Lockings')

# discriminant analysis
event_filters = [
    lambda x: type(x) == Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn == dtnn_types["Distractor"],
    lambda x: type(x) == Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT' and x.dtn == dtnn_types["Target"]]
event_names = ["Distractor", "Target"]
viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='Locked to Detected Fixation (using I-DT)')


plt.rcParams.update({'font.size': 22})
event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
# viz_pupil_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='RSVP ERP, locked to Detected Fixation (using Patch-Sim)')
viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='Locked to Detected Fixation (using I-DT)')
