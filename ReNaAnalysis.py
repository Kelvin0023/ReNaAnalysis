import json
import os

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

from rena.utils.data_utils import RNStream

# file paths
# data_path = 'C:/Recordings/11_17_2021_22_56_15-Exp_myexperiment-Sbj_someone-Ssn_0.dats'
from VarjoInterface import varjo_epochs_to_df
from utils import interpolate_array_nan, add_event_markers_to_data_array, generate_epochs, plot_epochs_visual_search, \
    visualize_epochs, generate_condition_sequence

tmin = -0.1
tmax = 3
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

trial_data_export_root = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/SingleTrials'

# participant_data_dict = {'AN': {
#     'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/11_13_2021_11_04_11-Exp_ReNaPilot-Sbj_AN-Ssn_0.dats',
#     'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/ReNaItemCatalog_11-13-2021-11-03-54.json',
#     'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/ReNaSessionLog_11-13-2021-11-03-54.json'},
#     'ZL': {
#         'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/11_10_2021_12_06_17-Exp_ReNaPilot-Sbj_ZL-Ssn_0.dats',
#         'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/ReNaItemCatalog_11-10-2021-12-04-46.json',
#         'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/ReNaSessionLog_11-10-2021-12-04-46.json'},
#     'WZ': {
#         'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-22-2021/11_22_2021_17_12_12-Exp_ReNa-Sbj_Pilot-WZ-Ssn_1.dats',
#         'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-22-2021/ReNaItemCatalog_11-22-2021-17-10-52.json',
#         'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-22-2021/ReNaSessionLog_11-22-2021-17-10-52.json'}}
participant_data_dict = {'AN': {
    'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/12-16-2021/12_16_2021_15_40_16-Exp_ReNa-Sbj_AN-Ssn_2.dats',
    'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/12-16-2021/ReNaItemCatalog_12-16-2021-15-40-01.json',
    'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/12-16-2021/ReNaSessionLog_12-16-2021-15-40-01.json'}}

varjoEyetracking_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeData.json'
varjoEyetracking_preset = json.load(open(varjoEyetracking_preset_path))
varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']
# varjoEyetracking_channelNames = ["L Pupil Diameter", "R Pupil Diameter",
#                                  "L Gaze Direction X", "L Gaze Direction Y", "L Gaze Direction Z",
#                                  "R Gaze Direction X", "R Gaze Direction Y", "R Gaze Direction Z",
#                                  "L Gaze Origin X", "L Gaze Origin Y", "L Gaze Origin Z",
#                                  "R Gaze Origin X", "R Gaze Origin Y", "R Gaze Origin Z",
#                                  "Gaze Combined Direction X", "Gaze Combined Direction Y", "Gaze Combined Direction Z",
#                                  "Gaze Combined Origin X", "Gaze Combined Origin Y", "Gaze Combined Origin Z",
#                                  "Focal Distance", "Focal Stability"
#                                  ]

title = 'ReNaPilot 2021'
event_ids = {'BlockBegins': 4, 'Novelty': 3, 'Target': 2, 'Distractor': 1}

epochs_pupil_rsvp = None
epochs_pupil_carousel = None

for participant_index, participant_code_data_path_dict in enumerate(participant_data_dict.items()):
    participant_code, participant_data_path_dict = participant_code_data_path_dict

    data_path = participant_data_path_dict['data_path']
    item_catalog_path = participant_data_path_dict['item_catalog_path']
    session_log_path = participant_data_path_dict['session_log_path']
    # process code after this
    data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
    item_catalog = json.load(open(item_catalog_path))
    session_log = json.load(open(session_log_path))

    item_codes = list(item_catalog.values())

    # process data  # TODO iterate over conditions
    event_markers_rsvp = data['Unity.ReNa.EventMarkers'][0][0:4]
    event_markers_carousel = data['Unity.ReNa.EventMarkers'][0][4:8]
    event_markers_vs = data['Unity.ReNa.EventMarkers'][0][8:12]

    event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]

    itemMarkers = data['Unity.ReNa.ItemMarkers'][0]
    itemMarkers_timestamps = data['Unity.ReNa.ItemMarkers'][1]

    eyetracking_data = data['Unity.VarjoEyeTracking'][0]
    eyetracking_data_timestamps = data['Unity.VarjoEyeTracking'][1]

    '''
    create blocked sequence data
    '''
    block_sequences_RSVP = generate_condition_sequence(
        event_markers_rsvp, event_markers_timestamps, eyetracking_data, eyetracking_data_timestamps,
        varjoEyetracking_channelNames,
        session_log,
        item_codes,
        srate=200)
    # Export the block sequences
    for block_index, bs in enumerate(block_sequences_RSVP):
        trial_export_path = os.path.join(trial_data_export_root, str(participant_index + 1), str(block_index + 1))
        os.makedirs(trial_export_path, exist_ok=True)

    '''  create the epoched adata
    epochs_pupil_rsvp_this_participant, epochs_rsvp_gaze_this_participant = generate_epochs(event_markers_rsvp,
                                                                                            event_markers_timestamps,
                                                                                            eyetracking_data,
                                                                                            eyetracking_data_timestamps,
                                                                                            varjoEyetracking_channelNames,
                                                                                            session_log,
                                                                                            item_codes, tmin, tmax,
                                                                                            event_ids, color_dict,
                                                                                            title='{0}, Participant {1}, Condition {2}'.format(
                                                                                                title,
                                                                                                participant_code,
                                                                                                'RSVP'),
                                                                                            is_plotting=False)
    epochs_carousel_this_participant, epochs_carousel_gaze_this_participant = generate_epochs(event_markers_carousel,
                                                                                              event_markers_timestamps,
                                                                                              eyetracking_data,
                                                                                              eyetracking_data_timestamps,
                                                                                              varjoEyetracking_channelNames,
                                                                                              session_log,
                                                                                              item_codes, tmin, tmax,
                                                                                              event_ids, color_dict,
                                                                                              title='{0}, Participant {1}, Condition {2}'.format(
                                                                                                  title,
                                                                                                  participant_code,
                                                                                                  'Carousel'),
                                                                                              is_plotting=False)
    epochs_pupil_rsvp = epochs_pupil_rsvp_this_participant if epochs_pupil_rsvp is None else mne.concatenate_epochs(
        [epochs_pupil_rsvp, epochs_pupil_rsvp_this_participant])
    epochs_pupil_carousel = epochs_carousel_this_participant if epochs_pupil_carousel is None else mne.concatenate_epochs(
        [epochs_pupil_carousel, epochs_carousel_this_participant])
    '''

    # plot_epochs_visual_search(itemMarkers, itemMarkers_timestamps, event_markers_vs, event_markers_timestamps,
    #                           eyetracking_data,
    #                           eyetracking_data_timestamps,
    #                           varjoEyetracking_preset['ChannelNames'], session_log,
    #                           item_codes, tmin, tmax, color_dict, title=title + ' Carousel')

    ''' Export the per-trial epochs for gaze behavior analysis
    epochs_carousel_gaze_this_participant_trial_dfs = varjo_epochs_to_df(epochs_carousel_gaze_this_participant.copy())
    for trial_index, single_trial_df in enumerate(epochs_carousel_gaze_this_participant_trial_dfs):
        trial_export_path = os.path.join(trial_data_export_root, str(participant_index + 1), str(trial_index + 1))
        os.makedirs(trial_export_path, exist_ok=True)
        fn = 'varjo_gaze_output_single_trial_participant_{0}_{1}.csv'.format(participant_index + 1, trial_index + 1)
        single_trial_df.reset_index()
        single_trial_df.to_csv(os.path.join(trial_export_path, fn), index=False)
    '''


# visualize_epochs(epochs_rsvp, event_ids, tmin, tmax, color_dict, '{0}, Averaged across Participants, Condition {
# 1}'.format(title, 'RSVP')) visualize_epochs(epochs_carousel, event_ids, tmin, tmax, color_dict, '{0},
# Averaged across Participants, Condition {1}'.format(title, 'Carousel'))

