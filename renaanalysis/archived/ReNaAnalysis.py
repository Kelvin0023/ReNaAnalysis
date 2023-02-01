import json
import os

import mne
import numpy as np

from rena.utils.data_utils import RNStream

# file paths
# data_path = 'C:/Recordings/11_17_2021_22_56_15-Exp_myexperiment-Sbj_someone-Ssn_0.dats'
from renaanalysis.utils.utils import generate_pupil_event_epochs, \
    visualize_pupil_epochs

data_root = "C:/Users/S-Vec/Dropbox/ReNa/data"

'''#################################################################################################
Path to store the full block data

'''
trial_data_export_root = 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/SingleTrials'
block_data_export_root = {'RSVP': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/Blocks-RSVP',
                          'Carousel': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/Blocks-Carousel',
                          'VS': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/Blocks-VS',
                          'TS': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/Blocks-TS'}
#################################################################################################

# Only put interested conditions here
event_marker_condition_index_dict = {'RSVP': slice(0, 4),
                                     'Carousel': slice(4, 8),
                                     # 'VS': slice(8, 12),
                                     # 'TS': slice(12, 16)
                                     }
tmin = -0.1
tmax = 3
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

participant_data_dict = {

    'WZ': {
        'data_path': 'ReNaPilot-2021Fall/11-22-2021/11_22_2021_17_12_12-Exp_ReNa-Sbj_Pilot-WZ-Ssn_1.dats',
        'item_catalog_path': 'ReNaPilot-2021Fall/11-22-2021/ReNaItemCatalog_11-22-2021-17-10-52.json',
        'session_log_path': 'ReNaPilot-2021Fall/11-22-2021/ReNaSessionLog_11-22-2021-17-10-52.json'},

    'JP': {  # 1/31/2022
        'data_path': 'ReNaPilot-2022Spring/02-15-2022/02_15_2022_17_23_57-Exp_ReNaPilot-Sbj_js-Ssn_0.dats',
        'item_catalog_path': 'ReNaPilot-2022Spring/02-15-2022/ReNaItemCatalog_02-15-2022-17-23-25.json',
        'session_log_path': 'ReNaPilot-2022Spring/02-15-2022/ReNaSessionLog_02-15-2022-17-23-25.json'},

    'ZL': {  # 1/31/2022
    'data_path': 'ReNaPilot-2022Spring/01-31-2022/01_31_2022_15_10_12-Exp_ReNa-Sbj_ZL-Ssn_0.dats',
    'item_catalog_path': 'ReNaPilot-2022Spring/01-31-2022/ReNaItemCatalog_01-31-2022-15-09-45.json',
    'session_log_path': 'ReNaPilot-2022Spring/01-31-2022/ReNaSessionLog_01-31-2022-15-09-45.json'},

    'AN': {  # 12/16/2022
        'data_path': 'ReNaPilot-2021Fall/12-16-2021/12_16_2021_15_40_16-Exp_ReNa-Sbj_AN-Ssn_2.dats',
        'item_catalog_path': 'ReNaPilot-2021Fall/12-16-2021/ReNaItemCatalog_12-16-2021-15-40-01.json',
        'session_log_path': 'ReNaPilot-2021Fall/12-16-2021/ReNaSessionLog_12-16-2021-15-40-01.json'}
}

# newest eyetracking data channel format
varjoEyetrackingComplete_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'
# old eyetracking data channel format
varjoEyetracking_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeData.json'

title = 'ReNaPilot 2022'
# event_ids = {'BlockBegins': 4, 'Novelty': 3, 'Target': 2, 'Distractor': 1}
event_ids = {'Novelty': 3, 'Target': 2, 'Distractor': 1}  # event_ids_for_interested_epochs

# end of setup parameters, start of the main block ######################################################
participant_data_dict = dict(
    [(p, dict([(x, os.path.join(data_root, y)) for x, y in d.items()])) for p, d in participant_data_dict.items()])
condition_epochs_pupil_dict = dict([(c, None) for c in event_marker_condition_index_dict.keys()])
condition_event_label_dict = dict([(c, np.empty(0)) for c in event_marker_condition_index_dict.keys()])

for participant_index, participant_code_data_path_dict in enumerate(participant_data_dict.items()):
    participant_code, participant_data_path_dict = participant_code_data_path_dict
    print("Working on participant {0}, {1} of {2}".format(participant_code, participant_index, len(participant_data_dict)))
    data_path = participant_data_path_dict['data_path']
    item_catalog_path = participant_data_path_dict['item_catalog_path']
    session_log_path = participant_data_path_dict['session_log_path']
    # process code after this
    data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
    item_catalog = json.load(open(item_catalog_path))
    session_log = json.load(open(session_log_path))

    item_codes = list(item_catalog.values())
    event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]
    itemMarkers = data['Unity.ReNa.ItemMarkers'][0]
    itemMarkers_timestamps = data['Unity.ReNa.ItemMarkers'][1]

    # check whether the data has the new or old eye data format
    # if 'Unity.VarjoEyeTrackingComplete' in data.keys():
    eyetracking_data_timestamps = data['Unity.VarjoEyeTrackingComplete'] if 'Unity.VarjoEyeTrackingComplete' in data.keys() else data['Unity.VarjoEyeTracking']
    eyetracking_data = eyetracking_data_timestamps[0]
    eyetracking_timestamps = eyetracking_data_timestamps[1]
    if len(eyetracking_data_timestamps[0]) == 34:
        varjoEyetracking_preset = json.load(open(varjoEyetrackingComplete_preset_path))
        varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']
    elif len(eyetracking_data_timestamps[0]) == 22:
        # if we are using the old eye data channels, change the name of pupil channels to the new
        varjoEyetracking_preset = json.load(open(varjoEyetracking_preset_path))
        varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']
        varjoEyetracking_channelNames[varjoEyetracking_channelNames.index('L Pupil Diameter')] = 'left_pupil_size'
        varjoEyetracking_channelNames[varjoEyetracking_channelNames.index('R Pupil Diameter')] = 'right_pupil_size'
    else:
        raise Exception("Invalid eye data")
    # process data
    for condition_name, condition_event_marker_index in event_marker_condition_index_dict.items():
        print("Processing Condition {0} for participant {1}".format(condition_name, participant_code))
        event_markers = data['Unity.ReNa.EventMarkers'][0][condition_event_marker_index]

        ''' create whole block sequences
        block_sequences = generate_condition_sequence(
            event_markers, event_markers_timestamps, eyetracking_data, eyetracking_data_timestamps,
            varjoEyetracking_channelNames,
            session_log,
            item_codes,
            srate=200)
        for block_index, bs in enumerate(block_sequences):
            block_export_path = os.path.join(block_data_export_root[condition_name], str(participant_index + 1), str(block_index + 1))
            os.makedirs(block_export_path, exist_ok=True)
            fn = 'varjo_gaze_output_single_block_Carousel_participant_{0}_block_{1}.csv'.format(participant_index + 1,
                                                                                                block_index + 1)
            df = varjo_block_seq_to_df(bs)
            df.reset_index()
            df.to_csv(os.path.join(block_export_path, fn), index=False)
        '''

        '''  create the epoched adata'''
        _epochs_pupil, _event_labels = generate_pupil_event_epochs(event_markers,
                                                                   event_markers_timestamps,
                                                                   eyetracking_data,
                                                                   eyetracking_timestamps,
                                                                   varjoEyetracking_channelNames,
                                                                   session_log,
                                                                   item_codes, tmin, tmax,
                                                                   event_ids, color_dict,
                                                                   title='{0}, Participant {1}, Condition {2}'.format(
                                                                 title,
                                                                 participant_code,
                                                                 condition_name),
                                                                   is_plotting=True)

        condition_epochs_pupil_dict[condition_name] = _epochs_pupil if condition_epochs_pupil_dict[
                                                                           condition_name] is None else mne.concatenate_epochs(
            [condition_epochs_pupil_dict[condition_name], _epochs_pupil])
        condition_event_label_dict[condition_name] = np.concatenate([condition_event_label_dict[condition_name], _event_labels])
        pass
    ''' Export the per-trial epochs for gaze behavior analysis
    epochs_carousel_gaze_this_participant_trial_dfs = varjo_epochs_to_df(epochs_carousel_gaze_this_participant.copy())
    for trial_index, single_trial_df in enumerate(epochs_carousel_gaze_this_participant_trial_dfs):
        trial_export_path = os.path.join(trial_data_export_root, str(participant_index + 1), str(trial_index + 1))
        os.makedirs(trial_export_path, exist_ok=True)
        fn = 'varjo_gaze_output_single_trial_participant_{0}_{1}.csv'.format(participant_index + 1, trial_index + 1)
        single_trial_df.reset_index()
        single_trial_df.to_csv(os.path.join(trial_export_path, fn), index=False)
    '''
''' Export per-condition pupil epochs 
for condition_name in event_marker_condition_index_dict.keys():
    trial_x_export_path = os.path.join(trial_data_export_root, "epochs_pupil_raw_condition_{0}.npy".format(condition_name))
    trial_y_export_path = os.path.join(trial_data_export_root, "epoch_labels_pupil_raw_condition_{0}".format(condition_name))
    np.save(trial_x_export_path, condition_epochs_pupil_dict[condition_name].get_data())
    np.save(trial_y_export_path, condition_event_label_dict[condition_name])
'''


visualize_pupil_epochs(condition_epochs_pupil_dict['Carousel'], event_ids, tmin, tmax, color_dict,
                 '{0}, Averaged across Participants, Condition {1}'.format(title, 'Carousel'))
