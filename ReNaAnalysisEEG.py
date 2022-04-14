import json
import os
import pickle
import time
from collections import defaultdict
import numpy as np
import mne
from rena.utils.data_utils import RNStream

from utils import generate_pupil_event_epochs, \
    flatten_list, generate_eeg_event_epochs, visualize_pupil_epochs, visualize_eeg_epochs, \
    read_file_lines_as_list

#################################################################################################
is_data_preloaded = True
is_epochs_preloaded = False
is_regenerate_ica = True
is_save_loaded_data = True

preloaded_dats_path = 'participant_session_dict.p'
data_root = "C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects"
epoch_data_export_root = "C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects_Epochs"
eventMarker_conditionIndex_dict = {
                                    'RSVP': slice(0, 4),
                                   'Carousel': slice(4, 8),
                                   # 'VS': slice(8, 12),
                                   # 'TS': slice(12, 16)
                                   }  # Only put interested conditions here
tmin_pupil = -0.1
tmax_pupil = 3.
tmin_eeg = -.5
tmax_eeg = 1.2
# eeg_picks = ['P4', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
eeg_picks = ['P4', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
# eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names

color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

# newest eyetracking data channel format
varjoEyetrackingComplete_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'

event_ids = {'Novelty': 3, 'Target': 2, 'Distractor': 1}  # event_ids_for_interested_epochs

# end of setup parameters, start of the main block ######################################################
start_time = time.time()
participant_list = os.listdir(data_root)
participant_directory_list = [os.path.join(data_root, x) for x in participant_list]

participant_session_dict = defaultdict(dict)  # create a dict that holds participant -> sessions -> list of sessionFiles
participant_condition_epoch_dict = defaultdict(dict)
participant_badchannel_dict = dict()
# create a dict that holds participant -> condition epochs
for participant, participant_directory in zip(participant_list, participant_directory_list):
    file_names = os.listdir(participant_directory)
    # assert len(file_names) % 3 == 0
    # must have #files divisible by 3. That is, we have a itemCatalog, SessionLog and data file for each experiment session.
    num_sessions = flatten_list([[int(s) for s in txt if s.isdigit()] for txt in file_names])
    num_sessions = len(np.unique(num_sessions))
    if os.path.exists(os.path.join(participant_directory, 'badchannels.txt')):  # load bad channels for this participant
        participant_badchannel_dict[participant] = read_file_lines_as_list(os.path.join(participant_directory, 'badchannels.txt'))
    for i in range(num_sessions):
        participant_session_dict[participant][i] = [os.path.join(participant_directory, x) for
                                                    x in ['{0}.dats'.format(i),
                                                          '{0}_ReNaItemCatalog.json'.format(i),
                                                          '{0}_ReNaSessionLog.json'.format(i),
                                                          '{0}_ParticipantSessionICA'.format(
                                                              i)]]  # file path for ICA solution and

# preload all the .dats
if not is_epochs_preloaded:
    if not is_data_preloaded:
        print("Preloading .dats")  # TODO parallelize loading of .dats
        for participant_index, session_dict in participant_session_dict.items():
            print("Working on participant-code[{0}] of {1}".format(int(participant_index), len(participant_session_dict) - 1))
            for session_index, session_files in session_dict.items():
                print("Session {0} of {1}".format(session_index + 1, len(session_dict)))
                data_path, item_catalog_path, session_log_path, session_ICA_path = session_files
                if os.path.exists(
                        data_path.replace('dats', 'p')):  # load pickle if it's available as it is faster than dats
                    data = pickle.load(open(data_path.replace('dats', 'p'), 'rb'))
                else:
                    data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
                participant_session_dict[participant_index][session_index][0] = data
        # save the preloaded .dats
        if is_save_loaded_data:
            print("Saving preloaded sessions...")
            pickle.dump(participant_session_dict, open(preloaded_dats_path, 'wb'))
    else:
        print("Loading preloaded sessions...")
        participant_session_dict = pickle.load(open('participant_session_dict.p', 'rb'))

    dats_loading_end_time = time.time()
    print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))

    for participant_index, session_dict in participant_session_dict.items():
        # print("Working on participant {0} of {1}".format(int(participant_index) + 1, len(participant_session_dict)))
        for session_index, session_files in session_dict.items():
            data, item_catalog_path, session_log_path, session_ICA_path = session_files
            item_catalog = json.load(open(item_catalog_path))
            session_log = json.load(open(session_log_path))
            item_codes = list(item_catalog.values())

            # markers
            event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]
            # itemMarkers = data['Unity.ReNa.ItemMarkers'][0]
            # itemMarkers_timestamps = data['Unity.ReNa.ItemMarkers'][1]

            # data
            varjoEyetracking_preset = json.load(open(varjoEyetrackingComplete_preset_path))
            varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']

            eyetracking_timestamps = data['Unity.VarjoEyeTrackingComplete'][1]
            eyetracking_data = data['Unity.VarjoEyeTrackingComplete'][0]

            eeg_timestamps = data['BioSemi'][1]
            eeg_data = data['BioSemi'][0][1:65, :]  # take only the EEG channels
            ecg_data = data['BioSemi'][0][65:67, :]  # take only the EEG channels

            for condition_name, condition_event_marker_index in eventMarker_conditionIndex_dict.items():
                print("Processing Condition {0} for participant-code[{1}] of {2}, session {3} of {4}".format(condition_name,
                                                                                                       int(participant_index) + 1,
                                                                                                       len(participant_session_dict),
                                                                                                       session_index + 1,
                                                                                                       len(session_dict)))
                event_markers = data['Unity.ReNa.EventMarkers'][0][condition_event_marker_index]

                _epochs_pupil, _ = generate_pupil_event_epochs(event_markers,
                                                                           event_markers_timestamps,
                                                                           eyetracking_data,
                                                                           eyetracking_timestamps,
                                                                           varjoEyetracking_channelNames,
                                                                           session_log,
                                                                           item_codes, tmin_pupil, tmax_pupil,
                                                                           event_ids, color_dict,
                                                                           is_plotting=False)

                _epochs_eeg, _epochs_eeg_ICA_cleaned, _, _, _ = generate_eeg_event_epochs(
                    event_markers,
                    event_markers_timestamps,
                    eeg_data,
                    ecg_data,
                    eeg_timestamps,
                    session_log,
                    item_codes,
                    session_ICA_path,
                    tmin_eeg, tmax_eeg,
                    event_ids,
                    is_regenerate_ica,
                    bad_channels=participant_badchannel_dict[participant_index] if participant_index in participant_badchannel_dict.keys() else None)

                if condition_name not in participant_condition_epoch_dict[participant_index].keys():
                    participant_condition_epoch_dict[participant_index][condition_name] = (
                    _epochs_pupil, _epochs_eeg, _epochs_eeg_ICA_cleaned)
                else:
                    participant_condition_epoch_dict[participant_index][condition_name] = (
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][0], _epochs_pupil]),
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][1], _epochs_eeg]),
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][2],
                             _epochs_eeg_ICA_cleaned])
                    )

    if is_save_loaded_data: pickle.dump(participant_condition_epoch_dict,
                                        open('participant_condition_epoch_dict.p', 'wb'))
else:  # if epochs are preloaded and saved
    print("Loading preloaded epochs ...")
    participant_condition_epoch_dict = pickle.load(open('participant_condition_epoch_dict.p', 'rb'))
    dats_loading_end_time = time.time()
    print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))


# get all the epochs for conditions and plots per condition
print("Creating plots across all participants per condition")
for condition_name in eventMarker_conditionIndex_dict.keys():
    print("Creating plots for condition {0}".format(condition_name))
    condition_epoch_list = flatten_list([x.items() for x in participant_condition_epoch_dict.values()])
    condition_epochs = [e for c, e in condition_epoch_list if c == condition_name]
    condition_epochs_pupil = mne.concatenate_epochs([pupil for pupil, eeg, _ in condition_epochs])
    condition_epochs_eeg = mne.concatenate_epochs([eeg for pupil, eeg, _ in condition_epochs])
    condition_epochs_eeg_ica = mne.concatenate_epochs([eeg for pupil, eeg, eeg_ica in condition_epochs])
    title = 'Averaged across Participants, Condition {0}'.format(condition_name)
    # visualize_pupil_epochs(condition_epochs_pupil, event_ids, tmin_pupil, tmax_pupil, color_dict, title)
    visualize_eeg_epochs(condition_epochs_eeg_ica, event_ids, tmin_eeg, tmax_eeg, color_dict, eeg_picks, title + '. ICA Cleaned', is_plot_timeseries=True)
    visualize_eeg_epochs(condition_epochs_eeg, event_ids, tmin_eeg, tmax_eeg, color_dict, eeg_picks, title, is_plot_timeseries=True)


# get all the epochs and plots per participant
# for participant_index, condition_epoch_dict in participant_condition_epoch_dict.items():
#     for condition_name, condition_epochs in condition_epoch_dict.items():
#         condition_epochs_pupil = condition_epochs[0]
#         condition_epochs_eeg = condition_epochs[1]
#         condition_epochs_eeg_ica = condition_epochs[2]
#         title = 'Participants {0} - Condition {1}'.format(participant_index, condition_name)
#         # visualize_pupil_epochs(condition_epochs_pupil, event_ids, tmin_pupil, tmax_pupil, color_dict, title)
#         # visualize_eeg_epochs(condition_epochs_eeg_ica, event_ids, tmin_eeg, tmax_eeg, color_dict, eeg_picks,
#         #                      title + '. ICA Cleaned', is_plot_timeseries=True)
#         visualize_eeg_epochs(condition_epochs_eeg, event_ids, tmin_eeg, tmax_eeg, color_dict, eeg_picks, title, is_plot_timeseries=False, is_plot_topo_map=True)

end_time = time.time()
print("Took {0} seconds".format(end_time - start_time))


# condition_epochs_pupil_dict[condition_name] = _epochs_pupil if condition_epochs_pupil_dict[
#                                                                    condition_name] is None else mne.concatenate_epochs(
#     [condition_epochs_pupil_dict[condition_name], _epochs_pupil])
# condition_event_label_dict[condition_name] = np.concatenate(
#     [condition_event_label_dict[condition_name], _event_labels])
#         pass
''' Export the per-trial epochs for gaze behavior analysis
epochs_carousel_gaze_this_participant_trial_dfs = varjo_epochs_to_df(epochs_carousel_gaze_this_participant.copy())
for trial_index, single_trial_df in enumerate(epochs_carousel_gaze_this_participant_trial_dfs):
    trial_export_path = os.path.join(trial_data_export_root, str(participant_index + 1), str(trial_index + 1))
    os.makedirs(trial_export_path, exist_ok=True)
    fn = 'varjo_gaze_output_single_trial_participant_{0}_{1}.csv'.format(participant_index + 1, trial_index + 1)
    single_trial_df.reset_index()
    single_trial_df.to_csv(os.path.join(trial_export_path, fn), index=False)
'''

''' Export per-condition eeg-ica epochs with design matrices
'''
# for condition_name in eventMarker_conditionIndex_dict.keys():
#     condition_epoch_list = flatten_list([x.items() for x in participant_condition_epoch_dict.values()])
#     condition_epochs = [e for c, e in condition_epoch_list if c == condition_name]
#     condition_epochs_eeg_ica = mne.concatenate_epochs([eeg for pupil, eeg, eeg_ica in condition_epochs])
#
#     trial_x_export_path = os.path.join(epoch_data_export_root, "epochs_pupil_raw_condition_{0}.npy".format(condition_name))
#     trial_y_export_path = os.path.join(epoch_data_export_root, "epoch_labels_pupil_raw_condition_{0}".format(condition_name))
#     np.save(trial_x_export_path, condition_epochs_pupil_dict[condition_name].get_data())
#     np.save(trial_y_export_path, condition_event_label_dict[condition_name])
