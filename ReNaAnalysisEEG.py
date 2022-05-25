"""
1 is distractor, 2 is target, 3 is novelty
4 and 5 are block starts and ends
6 and 7 encodes fixation and saccade onset respectively

6: fixation onset distractor
7: fixation onset target
8: fixation onset novelty
9: fixation onset null
9: saccade onset
"""

import json
import os
import pickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import mne
from rena.utils.data_utils import RNStream

from eyetracking import gaze_event_detection
from fs_utils import load_participant_session_dict
from utils import generate_pupil_event_epochs, \
    flatten_list, generate_eeg_event_epochs, visualize_pupil_epochs, visualize_eeg_epochs, \
    read_file_lines_as_list, add_gaze_em_to_data, add_em_ts_to_data, rescale_merge_exg, create_gaze_behavior_events, \
    extract_block_data, find_fixation_saccade_targets

#################################################################################################
is_data_preloaded = False
is_epochs_preloaded = False
is_regenerate_ica = False
is_save_loaded_data = True

preloaded_dats_path = 'Data/participant_session_dict.p'
preloaded_epoch_path = 'Data/participant_condition_epoch_dict_RCV_fsrp.p'
preloaded_block_path = 'Data/participant_condition_block_dict.p'
base_root = "C:/Users/Lab-User/Dropbox/ReNa/Data/ReNaPilot-2022Spring/"
# base_root = "C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/"
varjoEyetrackingComplete_preset_path = 'C:/Users/Lab-User/PycharmProjects/rena_jp/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'
# varjoEyetrackingComplete_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'

data_root = os.path.join(base_root, "Subjects")
epoch_data_export_root = os.path.join(base_root, 'Subjects-Epochs')
eventMarker_conditionIndex_dict = {
    'RSVP': slice(0, 4),
    'Carousel': slice(4, 8),
    'VS': slice(8, 12),
    'TS': slice(12, 16)
}  # Only put interested conditions here
# FixationLocking_conditions = ['RSVP', 'Carousel', 'VS', 'TS']

tmin_pupil = -0.1
tmax_pupil = 3.
tmin_pupil_viz = -0.1
tmax_pupil_viz = 3.

tmin_eeg = -1.2
tmax_eeg = 2.4

# tmin_eeg_viz = tmin_eeg
# tmax_eeg_viz = tmax_eeg
tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.2

eyetracking_srate = 200
exg_srate = 2048

# eeg_picks = ['P4', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
eeg_picks = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
# eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names

color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green',
              'Fixation': 'blue', 'Saccade': 'orange',
              'FixationDistractor': 'blue', 'FixationTarget': 'red', 'FixationNovelty': 'green', 'FixationNull': 'grey'}
info_chns = ["info1", "info2", "info3"]
# newest eyetracking data channel format

# event_ids = {'Novelty': 3, 'Target': 2, 'Distractor': 1, }  # event_ids_for_interested_epochs
# locked_marker = 'GazeMarker'

# event_ids = {'Fixation': 6, 'Saccade': 7, }  # event_ids_for_interested_epochs
event_ids = {'FixationDistractor': 6, 'FixationTarget': 7, 'FixationNovelty': 8, 'FixationNull': 9, 'Saccade': 10}  # event_ids_for_interested_epochs
locked_marker = 'GazeBehavior'

eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
ecg_ch_name='ECG00'

# end of setup parameters, start of the main block ######################################################
start_time = time.time()
participant_list = os.listdir(data_root)
participant_directory_list = [os.path.join(data_root, x) for x in participant_list]

gaze_statistics_path = preloaded_epoch_path.strip('.p') + 'gaze_statistics' + '.p'
gaze_behavior_path = preloaded_epoch_path.strip('.p') + 'gaze_behavior' + '.p'
participant_session_dict = defaultdict(dict)  # create a dict that holds participant -> sessions -> list of sessionFiles
participant_condition_epoch_dict = defaultdict(dict)  # participants -> condition name -> epoch object
participant_condition_block_dict = defaultdict(dict)
condition_gaze_statistics = defaultdict(dict)
condition_gaze_behaviors = defaultdict(dict)
for condition_names in eventMarker_conditionIndex_dict.keys():
    condition_gaze_behaviors[condition_names]['fixations'] = []
    condition_gaze_behaviors[condition_names]['saccades'] = []
participant_badchannel_dict = dict()
# create a dict that holds participant -> condition epochs
for participant, participant_directory in zip(participant_list, participant_directory_list):
    file_names = os.listdir(participant_directory)
    # assert len(file_names) % 3 == 0
    # must have #files divisible by 3. That is, we have a itemCatalog, SessionLog and data file for each experiment session.
    num_sessions = flatten_list([[int(s) for s in txt if s.isdigit()] for txt in file_names])
    num_sessions = len(np.unique(num_sessions))
    if os.path.exists(os.path.join(participant_directory, 'badchannels.txt')):  # load bad channels for this participant
        participant_badchannel_dict[participant] = read_file_lines_as_list(
            os.path.join(participant_directory, 'badchannels.txt'))
    for i in range(num_sessions):
        participant_session_dict[participant][i] = [os.path.join(participant_directory, x) for
                                                    x in ['{0}.dats'.format(i),
                                                          '{0}_ReNaItemCatalog.json'.format(i),
                                                          '{0}_ReNaSessionLog.json'.format(i),
                                                          '{0}_ParticipantSessionICA'.format(
                                                              i)]]  # file path for ICA solution and

# preload all the .dats
if not is_epochs_preloaded:
    participant_session_dict = load_participant_session_dict(participant_session_dict, is_data_preloaded,
                                                             is_save_loaded_data, preloaded_dats_path)
    dats_loading_end_time = time.time()
    print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))

    for p_i, (participant_index, session_dict) in enumerate(participant_session_dict.items()):
        # print("Working on participant {0} of {1}".format(int(participant_index) + 1, len(participant_session_dict)))
        for session_index, session_files in session_dict.items():
            data, item_catalog_path, session_log_path, session_ICA_path = session_files
            item_catalog = json.load(open(item_catalog_path))
            session_log = json.load(open(session_log_path))
            item_codes = list(item_catalog.values())

            # markers
            event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]
            item_markers = data['Unity.ReNa.ItemMarkers'][0]
            item_marker_timestamps = data['Unity.ReNa.ItemMarkers'][1]

            # data
            varjoEyetracking_preset = json.load(open(varjoEyetrackingComplete_preset_path))
            varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']

            eyetracking_timestamps = data['Unity.VarjoEyeTrackingComplete'][1]
            eyetracking_data = data['Unity.VarjoEyeTrackingComplete'][0]

            exg_timestamps = data['BioSemi'][1]
            eeg_data = data['BioSemi'][0][1:65, :]  # take only the EEG channels
            ecg_data = data['BioSemi'][0][65:67, :]  # take only the EEG channels

            for condition_name, condition_event_marker_index in eventMarker_conditionIndex_dict.items():
                print("Processing Condition {0} for participant-code[{1}]: {5} of {2}, session {3} of {4}".format(
                    condition_name,
                    int(participant_index),
                    len(participant_session_dict),
                    session_index + 1,
                    len(session_dict), p_i + 1))
                event_markers = data['Unity.ReNa.EventMarkers'][0][condition_event_marker_index]

                # merge and rescale eeg and ecg
                exg_data = rescale_merge_exg(eeg_data, ecg_data)
                #########################
                #  detect the gaze events
                #########################

                # identify the events and add event array to the data array for both eyetracking and exg
                # add events
                data_eyetracking_em = add_em_ts_to_data(event_markers,
                                                        event_markers_timestamps,
                                                        eyetracking_data,
                                                        eyetracking_timestamps, session_log,
                                                        item_codes, eyetracking_srate)
                data_exg_em = add_em_ts_to_data(event_markers,
                                                event_markers_timestamps,
                                                exg_data,
                                                exg_timestamps, session_log,
                                                item_codes, exg_srate)

                # add gaze events
                data_eyetracking_egm, fixation_durations, normalized_fixation_count = add_gaze_em_to_data(
                    item_markers, item_marker_timestamps, event_markers,
                    event_markers_timestamps, data_eyetracking_em, session_log,
                    item_codes, eyetracking_srate, verbose=1)
                data_exg_egm, _, _ = add_gaze_em_to_data(
                    item_markers, item_marker_timestamps, event_markers,
                    event_markers_timestamps, data_exg_em, session_log,
                    item_codes, exg_srate, verbose=1)
                del data_eyetracking_em, data_exg_em

                # add gaze behaviors
                gaze_xy = eyetracking_data[
                    [varjoEyetracking_channelNames.index('gaze_forward_{0}'.format(x)) for x in ['x', 'y']]]
                gaze_status = eyetracking_data[varjoEyetracking_channelNames.index('status')]
                gaze_behavior_events, fixations, saccades = gaze_event_detection(gaze_xy, gaze_status,
                                                                                 eyetracking_timestamps)

                fixations = find_fixation_saccade_targets(fixations, saccades, eyetracking_timestamps, data_exg_egm)

                exg_gb_markers = create_gaze_behavior_events(fixations, saccades, eyetracking_timestamps, data_exg_egm[0])
                data_exg_egbm = np.concatenate([data_exg_egm, exg_gb_markers])

                eyetracking_gb_markers = create_gaze_behavior_events(fixations, saccades, eyetracking_timestamps, data_eyetracking_egm[0])
                data_eyetracking_egbm = np.concatenate([data_eyetracking_egm, eyetracking_gb_markers])
                del data_exg_egm, data_eyetracking_egm

                # create channels based on the event channels added
                exg_egbm_channles = ['LSLTimestamp'] + eeg_channel_names + [ecg_ch_name] + info_chns + ['EventMarker'] + ['GazeMarker'] + ["GazeBehavior"]
                exg_egbm_channle_types = ['misc'] + ['eeg'] * len(eeg_channel_names) + ['ecg'] + ['stim'] * 3 + ['stim'] * 3
                eyetracking_egbm_channels = ['LSLTimestamp'] + varjoEyetracking_channelNames + info_chns + ['EventMarker'] + ['GazeMarker'] + ["GazeBehavior"]
                eyetracking_egbm_channel_types = ['misc'] + ['misc'] * len(varjoEyetracking_channelNames) + ['stim'] * 3 + ['stim'] * 3

                #########################

                # generate the epochs
                _epochs_pupil, _ = generate_pupil_event_epochs(data_eyetracking_egbm,
                                                               eyetracking_egbm_channels, eyetracking_egbm_channel_types, tmin_pupil, tmax_pupil,
                                                               event_ids, locked_marker)

                _epochs_exg, _epochs_eeg_ICA_cleaned, labels_array, _, _ = generate_eeg_event_epochs(
                    data_exg_egbm,
                    exg_egbm_channles,
                    exg_egbm_channle_types,
                    session_ICA_path,
                    tmin_eeg, tmax_eeg,
                    event_ids, locked_marker,
                    is_regenerate_ica=is_regenerate_ica,
                    bad_channels=participant_badchannel_dict[
                        participant_index] if participant_index in participant_badchannel_dict.keys() else None)

                del data_eyetracking_egbm, data_exg_egbm
                #########################

                # extract block data
                # _blocks_eyetracking = extract_block_data(data_eyetracking_egbm, eyetracking_egbm_channels, eyetracking_srate)
                # _blocks_exg = extract_block_data(data_exg_egbm, exg_egbm_channles, exg_srate)

                # record gaze statistics
                if fixation_durations is not None and normalized_fixation_count is not None:
                    if 'durations' in condition_gaze_statistics[condition_name].keys():
                        condition_gaze_statistics[condition_name]['durations'] = dict([(event_type, duration_list +
                                                                                        condition_gaze_statistics[
                                                                                            condition_name][
                                                                                            'durations'][event_type])
                                                                                       for event_type, duration_list in
                                                                                       fixation_durations.items()])
                    else:
                        condition_gaze_statistics[condition_name]['durations'] = fixation_durations
                    if 'counts' in condition_gaze_statistics[condition_name].keys():
                        condition_gaze_statistics[condition_name]['counts'] = dict([(event_type, 0.5 * (
                                norm_count + condition_gaze_statistics[condition_name]['counts'][event_type])) for
                                                                                    event_type, norm_count in
                                                                                    normalized_fixation_count.items()])
                    else:
                        condition_gaze_statistics[condition_name]['counts'] = normalized_fixation_count
                # Add to gaze behaviors

                condition_gaze_behaviors[condition_name]['fixations'] = condition_gaze_behaviors[condition_name]['fixations'] + fixations
                condition_gaze_behaviors[condition_name]['saccades'] = condition_gaze_behaviors[condition_name]['saccades'] + saccades

                # Add the new epochs to the epoch dictionary
                if condition_name not in participant_condition_epoch_dict[participant_index].keys():
                    participant_condition_epoch_dict[participant_index][condition_name] = (
                        _epochs_pupil, _epochs_exg, _epochs_eeg_ICA_cleaned, labels_array)
                else:
                    participant_condition_epoch_dict[participant_index][condition_name] = (
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][0], _epochs_pupil]),
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][1], _epochs_exg]),
                        mne.concatenate_epochs(
                            [participant_condition_epoch_dict[participant_index][condition_name][2],
                             _epochs_eeg_ICA_cleaned]),
                        np.concatenate(
                            [participant_condition_epoch_dict[participant_index][condition_name][3], labels_array]
                        )
                    )
                # Add the new blocks to the block dictionary
                # if condition_name not in participant_condition_block_dict[participant_index].keys():
                #     participant_condition_block_dict[participant_index][condition_name] = (
                #         _blocks_eyetracking, _blocks_exg)
                # else:
                #     participant_condition_block_dict[participant_index][condition_name] = (
                #         participant_condition_block_dict[participant_index][condition_name][0] + _blocks_eyetracking,
                #         participant_condition_block_dict[participant_index][condition_name][1] + _blocks_exg
                #     )

                # continue to the next condition
            # continue to the next session
        # continue to the next participant

    if is_save_loaded_data:
        pickle.dump(participant_condition_epoch_dict, open(preloaded_epoch_path, 'wb'))
        # pickle.dump(participant_condition_block_dict, open(preloaded_epoch_path, 'wb'))
        pickle.dump(condition_gaze_statistics, open(gaze_statistics_path, 'wb'))

else:  # if epochs are preloaded and saved
    print("Loading preloaded epochs ...")
    participant_condition_epoch_dict = pickle.load(open(preloaded_epoch_path, 'rb'))
    dats_loading_end_time = time.time()
    print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))

# if condition_gaze_statistics is not None:
#     for condition_name in eventMarker_conditionIndex_dict.keys():
#         for event in event_ids.keys():
#             durations = np.array(condition_gaze_statistics[condition_name]['durations'][event.lower()])
#             durations = durations[durations < 1.4]
#             plt.hist(durations * 1e3, label=event, bins=20)
#             plt.legend()
#             plt.xlabel('Millisecond')
#             plt.ylabel('Count')
#             plt.xlim(0, 1500)
#             plt.ylim(0, 700)
#             plt.title('Fixation duration {0}-{1} (min = 141.4 ms)'.format(condition_name, event))
#             plt.show()
#     # plot counts
#     plt.rcParams["figure.figsize"] = (12.8, 7.2)
#     X = np.arange(3)
#     for i, condition_name in enumerate(eventMarker_conditionIndex_dict.keys()):
#         bar = plt.bar(X + 0.25 * i, [condition_gaze_statistics[condition_name]['counts'][event.lower()] for event in
#                                      event_ids.keys()], label=condition_name, width=0.25)
#         for rect in bar:
#             height = rect.get_height()
#             plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')
#
#     plt.xticks(np.linspace(0, 2.5, 3), event_ids.keys())
#     plt.legend()
#     plt.title('Normalized fixation counts across conditions and item types')
#     plt.show()
X = np.arange(3)
plt.rcParams["figure.figsize"] = (12.8, 7.2)
for i, condition_name in enumerate(eventMarker_conditionIndex_dict.keys()):
    # fixations = condition_gaze_behaviors[condition_name]['fixations']
    # plt.hist([f.duration for f in fixations if f.duration<4 and f.stim != 'null'], bins=20)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Count')
    # plt.title('Non-null Fixation Duration. Condition {0}'.format(condition_name))
    # plt.show()

    saccades = condition_gaze_behaviors[condition_name]['saccades']
    saccade_amplitudes = [s.amplitude for s in saccades if s.to_stim is not None and s.amplitude < 20 and s.peak_velocity < 700]
    saccade_peak_velocities = [s.peak_velocity for s in saccades if s.to_stim is not None and s.amplitude < 20 and s.peak_velocity < 700]

    plt.hist(saccade_amplitudes, bins=20)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Non-null designated Saccade Amplitude. Condition {0}'.format(condition_name))
    plt.show()
    #
    plt.hist(saccade_peak_velocities, bins=20)
    plt.xlabel('Degree/sec')
    plt.ylabel('Count')
    plt.title('Non-null designated Saccade Peak Velocity. Condition {0}'.format(condition_name))
    plt.show()

    plt.scatter(saccade_peak_velocities, saccade_amplitudes)
    plt.ylabel('Saccade Amplitude (Degree)')
    plt.xlabel('Saccade Peak Velocity (Deg/sec)')
    plt.title('Non-null designated Saccade Amplitude vs. Peak Velocity. Condition {0}'.format(condition_name))
    plt.show()
    # for stim in ['target', 'distractor', 'novelty']:
    #     fixations = condition_gaze_behaviors[condition_name]['fixations']
    #     plt.hist([f.duration for f in fixations if f.duration < 4 and f.stim == stim], bins=20)
    #     plt.xlabel('Time (sec)')
    #     plt.ylabel('Count')
    #     plt.title('{1} Fixation Duration. Condition {0}'.format(condition_name, stim))
    #     plt.show()
    # bar = plt.bar(X + 0.25 * i, [condition_gaze_statistics[condition_name]['counts'][event.lower()] for event in
    #                              event_ids.keys()], label=condition_name, width=0.25)


# get all the epochs for conditions and plots per condition
print("Creating plots across all participants per condition")
for condition_name in eventMarker_conditionIndex_dict.keys():
    print("Creating plots for condition {0}".format(condition_name))
    condition_epoch_list = flatten_list([x.items() for x in participant_condition_epoch_dict.values()])
    condition_epochs = [e for c, e in condition_epoch_list if c == condition_name]
    condition_epochs_pupil = mne.concatenate_epochs([pupil for pupil, _, _, _ in condition_epochs])
    # condition_epochs_eeg = mne.concatenate_epochs([eeg for pupil, eeg, _ in condition_epochs])
    condition_epochs_eeg_ica = mne.concatenate_epochs([eeg_ica for _, _, eeg_ica, _ in condition_epochs])
    title = 'Averaged across Participants, Condition {0}, {1} Locked'.format(condition_name, locked_marker)
    visualize_pupil_epochs(condition_epochs_pupil, event_ids, tmin_pupil_viz, tmax_pupil_viz, color_dict, title)
    visualize_eeg_epochs(condition_epochs_eeg_ica, event_ids, tmin_eeg_viz, tmax_eeg_viz, color_dict, eeg_picks,
                         title, is_plot_topo_map=True)
    del condition_epochs_pupil, condition_epochs_eeg_ica, condition_epochs
    # visualize_eeg_epochs(condition_epochs_eeg, event_ids, tmin_eeg, tmax_eeg, color_dict, eeg_picks, title,
    #                      is_plot_timeseries=True)

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
# print("exporting data")
# for condition_name in eventMarker_conditionIndex_dict.keys():
#     locking = 'FixationLocked' if condition_name in FixationLocking_conditions else 'EventLocked'
#     condition_epoch_list = flatten_list([x.items() for x in participant_condition_epoch_dict.values()])
#     condition_epochs = [e for c, e in condition_epoch_list if c == condition_name]
#     condition_epochs_eeg_ica: mne.Epochs = mne.concatenate_epochs([eeg_ica for pupil, eeg, eeg_ica, labels in condition_epochs])
#     condition_epochs_labels =  np.concatenate([labels for pupil, eeg, eeg_ica, labels in condition_epochs])
#
#     trial_x_export_path = os.path.join(epoch_data_export_root,
#                                        "epochs_{0}_eeg_ica_condition_{1}_data.npy".format(locking, condition_name))
#     trial_dm_export_path = os.path.join(epoch_data_export_root,
#                                        "epochs_{0}_eeg_ica_condition_{1}_DM.npy".format(locking, condition_name))
#     trial_y_export_path = os.path.join(epoch_data_export_root,
#                                        "epochs_{0}_eeg_ica_condition_{1}_labels.npy".format(locking, condition_name))
#     np.save(trial_x_export_path, condition_epochs_eeg_ica.copy().pick('eeg').get_data())
#     DM_picks = mne.pick_channels_regexp(condition_epochs_eeg_ica.ch_names, regexp=r'DM_.*')
#     np.save(trial_dm_export_path, condition_epochs_eeg_ica.copy().pick(DM_picks).get_data())
#     np.save(trial_y_export_path, condition_epochs_labels)

end_time = time.time()
print("Took {0} seconds".format(end_time - start_time))
