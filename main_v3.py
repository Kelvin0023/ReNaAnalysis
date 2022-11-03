

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
from eye.eyetracking import gaze_event_detection, gaze_event_detection_I_DT, gaze_event_detection_PatchSim
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
            data, item_catalog_path, session_log_path, session_ICA_path = session_files
            item_catalog = json.load(open(item_catalog_path))
            session_log = json.load(open(session_log_path))
            item_codes = list(item_catalog.values())

            # markers
            events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1], data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])

            # add gaze behaviors from I-DT
            events += gaze_event_detection_I_DT(data['Unity.VarjoEyeTrackingComplete'][0], data['Unity.VarjoEyeTrackingComplete'][1], events)
            # add gaze behaviors from patch sim
            events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)

            visualize_gaze_events(events, 6)
            rdf.add_participant_session(data, events, participant_index, session_index)

event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
colors = ['blue', 'red']
rdf.viz_pupil_epochs(["Distractor", "Target"], event_filters, colors)

# _epochs_pupil, _ = generate_pupil_event_epochs(data, events)
#
# _epochs_exg, _epochs_eeg_ICA_cleaned, labels_array, _, _ = generate_eeg_event_epochs(
#     data_exg_egbm,
#     exg_egbm_channles,
#     exg_egbm_channle_types,
#     session_ICA_path,
#     tmin_eeg, tmax_eeg,
#     event_ids_dict,
#     is_regenerate_ica=is_regenerate_ica,
#     bad_channels=participant_badchannel_dict[
#         participant_index] if participant_index in participant_badchannel_dict.keys() else None)
#
#
#             # extract block data
#             _blocks_eyetracking = extract_block_data(data_eyetracking_egbm, eyetracking_egbm_channels, eyetracking_srate, fixations, saccades)  # TODO move block visualization outside of the loop
#             # _blocks_exg = extract_block_data(data_exg_egbm, exg_egbm_channles, exg_srate, fixations, saccades)
#             del data_eyetracking_egbm, data_exg_egbm
#
#             # record gaze statistics
#             if fixation_durations is not None and normalized_fixation_count is not None:
#                 if 'durations' in condition_gaze_statistics[condition_name].keys():
#                     condition_gaze_statistics[condition_name]['durations'] = dict([(event_type, duration_list +
#                                                                                     condition_gaze_statistics[
#                                                                                         condition_name][
#                                                                                         'durations'][event_type])
#                                                                                    for event_type, duration_list in
#                                                                                    fixation_durations.items()])
#                 else:
#                     condition_gaze_statistics[condition_name]['durations'] = fixation_durations
#                 if 'counts' in condition_gaze_statistics[condition_name].keys():
#                     condition_gaze_statistics[condition_name]['counts'] = dict([(event_type, 0.5 * (
#                             norm_count + condition_gaze_statistics[condition_name]['counts'][event_type])) for
#                                                                                 event_type, norm_count in
#                                                                                 normalized_fixation_count.items()])
#                 else:
#                     condition_gaze_statistics[condition_name]['counts'] = normalized_fixation_count
#
#             # Add to gaze behaviors
#             condition_gaze_behaviors[condition_name]['fixations'] = condition_gaze_behaviors[condition_name]['fixations'] + fixations
#             condition_gaze_behaviors[condition_name]['saccades'] = condition_gaze_behaviors[condition_name]['saccades'] + saccades
#
#             # Add the new epochs to the epoch dictionary
#             if condition_name not in participant_condition_epoch_dict[participant_index].keys():
#                 participant_condition_epoch_dict[participant_index][condition_name] = (
#                     _epochs_pupil, _epochs_exg, _epochs_eeg_ICA_cleaned, labels_array)
#             else:
#                 participant_condition_epoch_dict[participant_index][condition_name] = (
#                     mne.concatenate_epochs(
#                         [participant_condition_epoch_dict[participant_index][condition_name][0], _epochs_pupil]),
#                     mne.concatenate_epochs(
#                         [participant_condition_epoch_dict[participant_index][condition_name][1], _epochs_exg]),
#                     mne.concatenate_epochs(
#                         [participant_condition_epoch_dict[participant_index][condition_name][2],
#                          _epochs_eeg_ICA_cleaned]),
#                     np.concatenate(
#                         [participant_condition_epoch_dict[participant_index][condition_name][3], labels_array]
#                     )
#                 )
#             # Add the new blocks to the block dictionary
#             # if condition_name not in participant_condition_block_dict[participant_index].keys():
#             #     participant_condition_block_dict[participant_index][condition_name] = (
#             #         _blocks_eyetracking, _blocks_exg)
#             # else:
#             #     participant_condition_block_dict[participant_index][condition_name] = (
#             #         participant_condition_block_dict[participant_index][condition_name][0] + _blocks_eyetracking,
#             #         participant_condition_block_dict[participant_index][condition_name][1] + _blocks_exg
#             #     )
#
#             # continue to the next condition
#             # continue to the next session
#         # continue to the next participant
#
#     if is_save_loaded_data:
#         pickle.dump(participant_condition_epoch_dict, open(preloaded_epoch_path, 'wb'))
#         # pickle.dump(participant_condition_block_dict, open(preloaded_epoch_path, 'wb'))
#         pickle.dump(condition_gaze_statistics, open(gaze_statistics_path, 'wb'))
#         pickle.dump(condition_gaze_behaviors, open(gaze_behavior_path, 'wb'))
#
# else:  # if epochs are preloaded and saved
#     print("Loading preloaded epochs ...")
#     participant_condition_epoch_dict = pickle.load(open(preloaded_epoch_path, 'rb'))
#     condition_gaze_statistics = pickle.load(open(gaze_statistics_path, 'rb'))
#     condition_gaze_behaviors = pickle.load(open(gaze_behavior_path, 'rb'))
#     dats_loading_end_time = time.time()
#     print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))

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

'''
X = np.arange(3)
plt.rcParams["figure.figsize"] = (12.8, 7.2)
for i, condition_name in enumerate(eventMarker_eventMarkerIndex_dict.keys()):
    # fixations = condition_gaze_behaviors[condition_name]['fixations']
    # plt.hist([f.duration for f in fixations if f.duration<4 and f.stim != 'null'], bins=20)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Count')
    # plt.title('Non-null Fixation Duration. Condition {0}'.format(condition_name))
    # plt.show()

    saccades = condition_gaze_behaviors[condition_name]['saccades']
    saccade_amplitudes = [s.amplitude for s in saccades if s.to_stim is not None and s.amplitude < 20 and s.peak_velocity < 700]
    saccade_peak_velocities = [s.peak_velocity for s in saccades if s.to_stim is not None and s.amplitude < 20 and s.peak_velocity < 700]
    saccade_peak_durations = [s.duration for s in saccades]

    plt.hist(saccade_amplitudes, bins=20)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Non-null designated Saccade Amplitude. Condition {0}'.format(condition_name))
    plt.show()

    plt.hist(saccade_peak_velocities, bins=20)
    plt.xlabel('Degree/sec')
    plt.ylabel('Count')
    plt.title('Non-null designated Saccade Peak Velocity. Condition {0}'.format(condition_name))
    plt.show()

    plt.hist(saccade_peak_durations, bins=20)
    plt.xlabel('Sec')
    plt.ylabel('Count')
    plt.title('Non-null designated Saccade Duration. Condition {0}'.format(condition_name))
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
'''

# get all the epochs for conditions and plots per condition
print("Creating plots across all participants per condition")
for condition_name in eventMarker_eventMarkerIndex_dict.keys():
    print("Creating plots for condition {0}".format(condition_name))
    condition_epoch_list = flatten_list([x.items() for x in participant_condition_epoch_dict.values()])
    condition_epochs = [e for c, e in condition_epoch_list if c == condition_name]
    condition_epochs_pupil = mne.concatenate_epochs([pupil for pupil, _, _, _ in condition_epochs])
    # condition_epochs_eeg = mne.concatenate_epochs([eeg for pupil, eeg, _ in condition_epochs])
    condition_epochs_eeg_ica = mne.concatenate_epochs([eeg_ica for _, _, eeg_ica, _ in condition_epochs])
    title = 'Averaged across Participants, Condition {0}, {1} Locked'.format(condition_name, event_viz)
    visualize_pupil_epochs(condition_epochs_pupil, event_ids_dict[event_viz], tmin_pupil_viz, tmax_pupil_viz, title)
    visualize_eeg_epochs(condition_epochs_eeg_ica, event_ids_dict[event_viz], tmin_eeg_viz, tmax_eeg_viz, eeg_picks,
                         title, is_plot_topo_map=True)

# get all the epochs and plots per participant
for participant_index, condition_epoch_dict in participant_condition_epoch_dict.items():
    for condition_name, condition_epochs in condition_epoch_dict.items():
        condition_epochs_pupil = condition_epochs[0]
        condition_epochs_eeg_ica = condition_epochs[2]
        title = 'Participants {0} - Condition {1}'.format(participant_index, condition_name)
        # visualize_pupil_epochs(condition_epochs_pupil, event_ids, tmin_pupil, tmax_pupil, color_dict, title)
        visualize_eeg_epochs(condition_epochs_eeg_ica, event_ids_dict[event_viz], tmin_eeg_viz, tmax_eeg_viz, eeg_picks, title, is_plot_timeseries=True, is_plot_topo_map=False, out_dir='figures')


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


