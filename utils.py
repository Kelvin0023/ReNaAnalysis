import math
import os
import random
from copy import copy

import numpy as np
import scipy
from mne.io import RawArray
from mne.preprocessing import create_ecg_epochs
from scipy.interpolate import interp1d
import json
import imageio

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

from rena.utils.data_utils import RNStream

from eyetracking import running_mean

FIXATION_MINIMAL_TIME = 1e-3 * 141.42135623730952
ITEM_TYPE_ENCODING = {1: 'distractor', 2: 'target', 3: 'novelty'}

START_OF_BLOCK_ENCODING = 4
END_OF_BLOCK_ENCODING = 5

def interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
    if np.sum(np.logical_not(not_nan)) / len(x) > 0.5:  # if more than half are nan
        raise ValueError("More than half of the given data array is nan")
    indices = np.arange(len(x))
    interp = interp1d(indices[not_nan], x[not_nan], fill_value="extrapolate")
    return interp(indices)


def interpolate_array_nan(data_array):
    """
    :param data_array: channel first, time last
    """
    return np.array([interpolate_nan(x) for x in data_array])


def interpolate_epochs_nan(epoch_array):
    """
    :param data_array: channel first, time last
    """
    rtn = []
    rejected_count = 0
    for e in epoch_array:
        temp = []
        try:
            for x in e:
                temp.append(interpolate_nan(x))
        except ValueError:  # something wrong with this epoch, maybe more than half are nan
            rejected_count += 1
            continue  # reject this epoch
        rtn.append(temp)
    print("Rejected {0} epochs of {1} total".format(rejected_count, len(epoch_array)))
    return np.array(rtn)


def interpolate_epoch_zeros(e):
    copy = np.copy(e)
    copy[copy == 0] = np.nan
    return interpolate_epochs_nan(copy)

def find_value_thresholding_interval(array, timestamps, value_threshold, time_threshold, time_tolerance=0.25):
    # change all zeros before the first non-zero entry to be nan
    array[:np.argwhere(array != 0)[0][0]] = np.nan
    below_threshold_index = np.argwhere(array < value_threshold)
    out = []
    for i in below_threshold_index[:, 0]:
        i_end = np.argmin(np.abs(timestamps-(timestamps[i] + time_threshold)))
        if np.all(array[i:i_end] < value_threshold):
            if (time_threshold - (timestamps[i_end] - timestamps[i])) / time_threshold < time_tolerance:
                i_peak = np.argmin(array[i:i_end])
                out.append((i, i_end, i + i_peak))
            else:
                print('exceed time tolerance ignoring interval')
    return out

def add_em_ts_to_data(event_markers, event_marker_timestamps, data_array, data_timestamps,
                      session_log,
                      item_codes, srate, pre_first_block_time=1, post_final_block_time=1):
    """
    add LSL timestamps, event markers based on the session log to the data array
    also discard data that falls other side the first and the last block
    :param event_markers:
    :param event_marker_timestamps:
    :param data_array:
    :param data_timestamps:
    :param session_log:
    :param item_codes:
    :param srate:
    :param pre_first_block_time:
    :param post_final_block_time:
    :return:
    """
    block_num = None
    assert event_markers.shape[0] == 4
    data_event_marker_array = np.zeros(shape=(4, data_array.shape[1]))
    first_block_start_index = None
    for i in range(event_markers.shape[1]):
        event, info1, info2, info3 = event_markers[:, i]
        data_event_marker_index = (np.abs(data_timestamps - event_marker_timestamps[i])).argmin()

        if str(int(event)) in session_log.keys():  # for start-of-block events
            # print('Processing block with ID: {0}'.format(event))
            block_num = event
            data_event_marker_array[3][data_event_marker_index] = START_OF_BLOCK_ENCODING  # encodes start of a block
            if first_block_start_index is None: first_block_start_index = data_event_marker_index
            continue
        elif event_markers[0, i - 1] != 0 and event == 0:  # this is the end of a block
            data_event_marker_array[3][data_event_marker_index] = END_OF_BLOCK_ENCODING  # encodes start of a block
            final_block_end_index = data_event_marker_index
            continue

        if event in item_codes:  # for item events
            targets = session_log[str(int(block_num))]['targets']
            distractors = session_log[str(int(block_num))]['distractors']
            novelties = session_log[str(int(block_num))]['novelties']

            if event in distractors:
                data_event_marker_array[3][data_event_marker_index] = 1
            elif event in targets:
                data_event_marker_array[3][data_event_marker_index] = 2
            elif event in novelties:
                data_event_marker_array[3][data_event_marker_index] = 3
            data_event_marker_array[:3, data_event_marker_index] = info1, info2, info3

    # remove the data before and after the last event
    out = np.concatenate([np.expand_dims(data_timestamps, axis=0), data_array, data_event_marker_array], axis=0)
    out = out[:, first_block_start_index - srate * pre_first_block_time:final_block_end_index + srate * post_final_block_time]
    return out

def add_design_matrix_to_data(data_array, event_marker_index, srate, erp_window, event_type_of_interest=(1, 2, 3)):
    '''
    expect data_array to have be of shape [time, LSLTimestamp(x 1)+data(x n)+event_markers(x n)]
    :param data_eventMarker_array_:
    :param event_type_of_interest: 1, 2, 3 only interested in targets distrctors and novelties
    :return:
    '''
    # get the event marker array
    eventMarker_array = data_array[event_marker_index]
    num_samples_erp_window = int(srate * (erp_window[1] - erp_window[0]))
    design_matrix = np.zeros((len(event_type_of_interest) * num_samples_erp_window, data_array.shape[1]))
    event_indices = [((flatten_list(np.argwhere(eventMarker_array == event_type)))) for event_type in event_type_of_interest]
    event_indices = flatten_list([[(e, event_type) for e in events] for event_type, events in zip(event_type_of_interest, event_indices)])
    for event_index, event_type in event_indices:
        dm_time_start_index = event_index
        dm_erpTime_start_index = (event_type - 1) * num_samples_erp_window
        for i in range(num_samples_erp_window):
            assert design_matrix[dm_erpTime_start_index+i, dm_time_start_index+i] == 0  # there cannot be overlapping events in the design matrix
            design_matrix[dm_erpTime_start_index+i, dm_time_start_index+i] = 1
    design_matrix_channel_names = flatten_list([['DM_E{0}_T{1}'.format(e_type, i) for i in range(num_samples_erp_window)] for e_type in event_type_of_interest])
    return np.concatenate([data_array, design_matrix], axis=0), design_matrix, design_matrix_channel_names


def add_gaze_em_to_data(item_markers, item_markers_timestamps, event_markers, event_marker_timestamps,
                        data_array,  # this data array already has timestamps, this function is called after add_em_ts_to_data
                        session_log, item_codes, srate, verbose, pre_block_time=1, post_block_time=1, foveate_value_threshold=15, foveate_duration_threshold=FIXATION_MINIMAL_TIME):
    block_list = []
    assert event_markers.shape[0] == 4
    data_event_marker_array = np.zeros(shape=(1, data_array.shape[1]))
    data_timestamps = data_array[0]

    item_block_start_indices = []
    item_block_end_indices = []

    for i in range(event_markers.shape[1]):
        event, info1, info2, info3 = event_markers[:, i]

        if str(int(event)) in session_log.keys():  # for start-of-block events
            # print('Processing block with ID: {0}'.format(event))
            block_list.append(event)
            item_block_start_indices.append(np.argmin(np.abs(item_markers_timestamps - event_marker_timestamps[i])))
            continue
        elif event_markers[0, i - 1] != 0 and event == 0:  # this is the end of a block
            item_block_end_indices.append(np.argmin(np.abs(item_markers_timestamps - event_marker_timestamps[i])))
            continue
    # iterate through blocks
    total_distractor_count = 0
    total_target_count = 0
    total_novelty_count = 0
    gazeRayIntersect_durations = {'distractor': [], 'target': [], 'novelty': []}

    for block_i, item_marker_block_start_index, item_marker_block_end_index in zip(block_list, item_block_start_indices, item_block_end_indices):
        targets = session_log[str(int(block_i))]['targets']
        distractors = session_log[str(int(block_i))]['distractors']
        novelties = session_log[str(int(block_i))]['novelties']

        # 2. find the nearest timestamp of the block start and end in the item marker timestamps
        item_markers_of_block = item_markers[:, item_marker_block_start_index:item_marker_block_end_index]
        item_markers_timestamps_of_block = item_markers_timestamps[
                                           item_marker_block_start_index:item_marker_block_end_index]

        item_block_start_indices.append(item_marker_block_start_index)
        item_block_end_indices.append(item_marker_block_end_index)

        for i in range(30):  # the item marker hold up to 30 items
            # this_item_marker = item_markers_of_block[i * 11: (i + 1) * 11, i::30]
            this_item_marker = item_markers_of_block[i * 11 : (i+1) * 11]
            this_item_code = this_item_marker[1, -1]  # TODO: change this to  np.max(this_item_marker[1, :]) after the 'reset item marker' update
            assert len(np.unique(this_item_marker[1, :])) == 2 or len(np.unique(this_item_marker[1, :])) == 1  # can only be either the item code or 0 if item is not active during that interval
            # check the item type
            if this_item_code in distractors:
                total_distractor_count += 1
                item_type = 'distractor'
                event_code = 1
            elif this_item_code in targets:
                total_target_count += 1
                item_type = 'target'
                event_code = 2
            elif this_item_code in novelties:
                total_novelty_count += 1
                item_type = 'novelty'
                event_code = 3
            else:
                # TODO: put the exception back after the 'reset item marker' update
                # raise Exception("Unknown item code {0} in block. This should NEVER happen!".format(this_item_code, block_i))
                if verbose: print('Out of block item found: should NOT happen again after RESET FIX')
                continue
            # find if there is gaze ray intersection
            # TODO: find saccade before fixations
            # foveate_indices = find_value_thresholding_interval(this_item_marker[2, :], item_markers_timestamps_of_block, foveate_value_threshold, foveate_duration_threshold)
            # force the first last to be not-intersected
            this_item_marker[4, 0] = 0
            this_item_marker[4, -1] = 0
            gaze_intersect_start_index = np.argwhere(np.diff(this_item_marker[4, :]) == 1)[:, 0]
            gaze_intersect_end_index = np.argwhere(np.diff(this_item_marker[4, :]) == -1)[:, 0]
            assert len(gaze_intersect_end_index) == len(gaze_intersect_start_index)

            # check if the intersects is long enough to warrant a fixation
            gaze_intersected_durations = item_markers_timestamps_of_block[gaze_intersect_end_index] - \
                                         item_markers_timestamps_of_block[gaze_intersect_start_index]
            # append_list_lines_to_file(gaze_intersected_durations, 'Data/FixationDurations')  # TODO: check this after we collect more data using the 'reset item marker' fix
            gazeRayIntersect_durations[item_type] += list(gaze_intersected_durations)

            true_fixations_indices = np.argwhere(gaze_intersected_durations > foveate_duration_threshold)[:, 0]
            true_fixation_timestamps = item_markers_timestamps_of_block[gaze_intersect_start_index[true_fixations_indices]]

            # find where in data marker to insert the marker
            data_event_marker_indices = [(np.abs(data_timestamps - x)).argmin() for x in true_fixation_timestamps]
            data_event_marker_array[0][data_event_marker_indices] = event_code
            # if len(true_fixation_timestamps) > 0: print(
            #     'Found {0} fixations for item {1} of type {2}, in block {3}'.format(len(true_fixation_timestamps),
            #                                                                         this_item_marker[1, 0],
            #                                                                         ITEM_TYPE_ENCODING[
            #                                                                             event_code], block_i))

        # 3. get the IsGazeRay intersected stream and their timestamps (item marker) keyed by the item count in block
        # 4. for each of the 30 items in the block, find where the IsGazeRay is true
        # 5. insert the gazed event marker in the data_event_marker_array at the data_timestamp nearest to the corresponding item_marker_timestamp

    # plt.plot(item_markers[1])
    # [plt.axvline(x=x, color='blue') for x in item_block_start_indices]
    # [plt.axvline(x=x, color='red') for x in item_block_end_indices]
    # plt.show()
    if verbose: print("found gaze ray intersects: %d distractors, %d targets, %d novelties" % (np.sum(data_event_marker_array[0]==1), np.sum(data_event_marker_array[0]==2), np.sum(data_event_marker_array[0]==3)))
    total_item_count = total_distractor_count + total_target_count + total_novelty_count
    normalized_fixation_count = {'distractor': np.sum(data_event_marker_array[0]==1) / total_distractor_count,
                        'target': np.sum(data_event_marker_array[0]==2) / total_target_count,
                        'novelty': np.sum(data_event_marker_array[0]==3) / total_novelty_count}
    return np.concatenate([data_array, data_event_marker_array], axis=0), gazeRayIntersect_durations, normalized_fixation_count,


def extract_block_data(_data, channel_names, srate, pre_block_time=.5, post_block_time=.5):  # event markers is the third last row
    # TODO add block end margins and use parameters block_end_margin_seconds
    block_starts = np.argwhere(_data[channel_names.index('EventMarker'), :] == 4) - int(pre_block_time * srate)# start of a block is denoted by event marker 4
    block_ends = np.argwhere(_data[channel_names.index('EventMarker'), :] == 5) + int(post_block_time * srate)  # end of a block is denoted by event marker 5

    block_sequences = [_data[:, i[0]:j[0]] for i, j in zip(block_starts, block_ends)]
    return block_sequences

    plt.rcParams["figure.figsize"] = (60, 15)
    em_color_code_dict = {1: 'b', 2: 'r', 3:'g', 4: 'black', 5: 'black'}
    gb_color_code_dict = {6: 'r', 7: 'g'}
    b = block_sequences[0]
    b = np.copy(b)
    t = b[0]
    p = b[channel_names.index('left_pupil_size'), :]
    p[p == 0] = np.nan
    p = interpolate_nan(p)
    xy = b[[channel_names.index('gaze_forward_x'), channel_names.index('gaze_forward_y')], :]
    xy_deg = (180 / math.pi) * np.arcsin(xy)
    dxy = np.diff(xy_deg, axis=1, prepend=xy_deg[:, :1])
    dtheta = np.linalg.norm(dxy, axis=0)
    velocities = dtheta / np.diff(t, prepend=1)
    velocities[0] = 0.

    plt.plot(t, p)
    for i, e in enumerate(b[channel_names.index('EventMarker'), :]):
        if e != 0:
            plt.axvline(t[i], alpha = 0.5, linewidth=2, color=em_color_code_dict[e])

    for i, e in enumerate(b[channel_names.index('GazeMarker'), :]):
        if e != 0:
            plt.axvline(t[i], linestyle='--', color='orange')
    plt.xlabel('Time (sec)')
    plt.title('Gaze ray intersects')
    plt.show()


    for i, e in enumerate(b[channel_names.index('EventMarker'), :]):
        if e != 0:
            plt.axvline(t[i], linewidth=2, color=em_color_code_dict[e])

    gb = b[channel_names.index('GazeBehavior'), :]
    previous_marker_index = None
    for i in range(len(gb) - 1):
        if gb[i] != 0:
            if previous_marker_index:
                plt.axvspan(t[previous_marker_index], t[i], alpha = 0.5, color=gb_color_code_dict[gb[i]])
            previous_marker_index = i
    plt.plot(t, velocities)
    plt.xlabel('Time (sec)')
    plt.title('Gaze ray intersects')
    plt.show()


    pass
    # block_sequences_resampled = []
    # # resample each block to be 100 Hz
    # for bs in block_sequences:  # don't resample the event marker sequences
    #     info = mne.create_info(['LSLTimestamp'] + data_channel_names + ['EventMarker', "info1", "info2", "info3"], sfreq=srate,
    #                            ch_types=['misc'] * (1 + len(data_channel_names)) + ['stim'] + ['misc'] * 3)
    #     raw = mne.io.RawArray(bs, info)
    #     raw_resampled = mne.io.RawArray(bs, info)  # resample to 100 Hz
    #     events = mne.find_events(raw, stim_channel='EventMarker')
    #     raw_resampled, events_resample = raw_resampled.resample(100, events=events)  # resample to 100 Hz
    #     raw_resampled.add_events(events_resample, stim_channel='EventMarker', replace=True)
    #     block_sequences_resampled.append(raw_resampled.get_data())

    return block_sequences  # a list of block sequences


def generate_pupil_event_epochs(data_, data_channels, data_channel_types, tmin, tmax, event_ids, locked_marker, erp_window=(.0, .8),srate=200,
                                verbose='WARNING'):  # use a fixed sampling rate for the sampling rate to match between recordings
    mne.set_log_level(verbose=verbose)

    info = mne.create_info(
        data_channels,
        sfreq=srate,
        ch_types=data_channel_types)  # with 3 additional info markers
    raw = mne.io.RawArray(data_, info)

    # only keep events that are in the block
    event_ids = dict([(event_name, event_code) for event_name, event_code in event_ids.items() if event_code in np.unique(mne.find_events(raw, stim_channel=locked_marker)[:, 2])])
    # pupil epochs
    epochs_pupil = Epochs(raw, events=find_events(raw, stim_channel=locked_marker), event_id=event_ids, tmin=tmin,
                          tmax=tmax,
                          baseline=(-0.1, 0.0),
                          preload=True,
                          verbose=False, picks=['left_pupil_size', 'right_pupil_size'])
    labels_array = epochs_pupil.events[:, 2]
    return epochs_pupil, labels_array


def rescale_merge_exg(data_array_EEG, data_array_ECG):
    data_array_EEG = data_array_EEG * 1e-6
    data_array_ECG = data_array_ECG * 1e-6
    data_array_ECG = (data_array_ECG[0] - data_array_ECG[1])[None, :]
    data_array = np.concatenate([data_array_EEG, data_array_ECG])
    return data_array

def generate_eeg_event_epochs(data_, data_channels, data_channle_types, ica_path, tmin, tmax, event_ids, locked_marker, erp_window=(.0, .8), srate=2048, verbose='CRITICAL',
                              is_regenerate_ica=False, lowcut=1, highcut=50., resample_srate=128, bad_channels=None):
    mne.set_log_level(verbose=verbose)
    biosemi_64_montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(
        data_channels,
        sfreq=srate,
        ch_types=data_channle_types)  # with 3 additional info markers and design matrix
    raw = mne.io.RawArray(data_, info)
    raw.set_montage(biosemi_64_montage)
    raw, _ = mne.set_eeg_reference(raw, 'average',
                                   projection=False)
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # mpl.use('Qt5Agg')
    # raw.plot(scalings='auto')
    if bad_channels:
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(method={'eeg': 'MNE'}, verbose='INFO')

    raw = raw.filter(l_freq=lowcut, h_freq=highcut)  # bandpass filter
    raw = raw.notch_filter(freqs=np.arange(60, 241, 60), filter_length='auto')
    raw = raw.resample(resample_srate)

    # recreate raw with design matrix
    data_array_with_dm, design_matrix, dm_ch_names = add_design_matrix_to_data(raw.get_data(), -4, resample_srate, erp_window=erp_window)

    info = mne.create_info(
        data_channels + dm_ch_names,
        sfreq=resample_srate,
        ch_types=data_channle_types + len(dm_ch_names) * ['stim'])  # with 3 additional info markers and design matrix
    raw = mne.io.RawArray(data_array_with_dm, info)
    raw.set_montage(biosemi_64_montage)

    # check if ica for this participant and session exists, create one if not
    if is_regenerate_ica or (not os.path.exists(ica_path + '.txt') or not os.path.exists(ica_path + '-ica.fif')):
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(raw, picks='eeg')
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG00', method='correlation',
                                                    threshold='auto')
        # ica.plot_scores(ecg_scores)
        if len(ecg_indices) > 0:
            [print(
                'Found ECG component at ICA index {0} with score {1}, adding to ICA exclude'.format(x, ecg_scores[x]))
             for x in ecg_indices]
            ica.exclude += ecg_indices
        else:
            print('No channel found to be significantly correlated with ECG, skipping auto ECG artifact removal')
        ica.plot_sources(raw)
        ica.plot_components()
        ica_excludes = input("Enter manual ICA components to exclude (use space to deliminate): ")
        if len(ica_excludes) > 0: ica.exclude += [int(x) for x in ica_excludes.split(' ')]
        f = open(ica_path + '.txt', "w")
        f.writelines("%s\n" % ica_comp for ica_comp in ica.exclude)
        f.close()
        ica.save(ica_path + '-ica.fif', overwrite=True)

        print('Saving ICA components', end='')
    else:
        ica = mne.preprocessing.read_ica(ica_path + '-ica.fif')
        with open(ica_path + '.txt', 'r') as filehandle:
            ica.exclude = [int(line.rstrip()) for line in filehandle.readlines()]

        print('Found and loaded existing ICA file', end='')

    print(': ICA exlucde component {0}'.format(str(ica.exclude)))
    raw_ica_recon = raw.copy()
    ica.apply(raw_ica_recon)
    # raw.plot(scalings='auto')
    # reconst_raw.plot(show_scrollbars=False, scalings='auto')

    reject = dict(eeg=600.)  # DO NOT reject or we will have a mismatch between EEG and pupil
    # only keep events that are in the block
    event_ids = dict([(event_name, event_code) for event_name, event_code in event_ids.items() if event_code in np.unique(mne.find_events(raw, stim_channel=locked_marker, shortest_event=1)[:, 2])])  # we may not have all target, distractor and novelty, especially in free-viewing
    epochs = Epochs(raw, events=find_events(raw, stim_channel=locked_marker, shortest_event=1), event_id=event_ids, tmin=tmin,
                    tmax=tmax,
                    baseline=(-0.1, 0.0),
                    preload=True,
                    verbose=False,
                    reject=reject)

    epochs_ICA_cleaned = Epochs(raw_ica_recon, events=find_events(raw, stim_channel=locked_marker, shortest_event=1), event_id=event_ids,
                                tmin=tmin,
                                tmax=tmax,
                                baseline=(-0.1, 0.0),
                                preload=True,
                                verbose=False,
                                reject=reject)

    labels_array = epochs.events[:, 2]
    return epochs, epochs_ICA_cleaned, labels_array, raw, raw_ica_recon


def visualize_pupil_epochs(epochs, event_ids, tmin, tmax, color_dict, title, srate=200, verbose='INFO', fig_size=(25.6, 14.4)):
    plt.rcParams["figure.figsize"] = fig_size
    mne.set_log_level(verbose=verbose)
    # epochs = epochs.apply_baseline((0.0, 0.0))
    for event_name, event_marker_id in event_ids.items():
        try:
            y = epochs[event_name].get_data()
        except KeyError:  # meaning this event does not exist in these epochs
            continue
        y = interpolate_epoch_zeros(y)  # remove nan
        y = interpolate_epochs_nan(y)  # remove nan
        assert np.sum(np.isnan(y)) == 0
        if len(y) == 0:
            print("visualize_pupil_epochs: all epochs bad, skipping {0}".format(event_name))
            continue
        y = np.mean(y, axis=1)  # average left and right
        y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

        y_mean = np.mean(y, axis=0)
        y_mean = y_mean - y_mean[int(abs(tmin) * srate)]  # baseline correct
        y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
        y2 = y_mean - scipy.stats.sem(y, axis=0)  # this is the lower envelope

        time_vector = np.linspace(tmin, tmax, y.shape[-1])
        plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                         interpolate=True,
                         alpha=0.5)
        plt.plot(time_vector, y_mean, c=color_dict[event_name],
                 label='{0}, N={1}'.format(event_name, epochs[event_name].get_data().shape[0]))

    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
    plt.legend()
    plt.title(title)
    plt.show()


def visualize_eeg_epochs(epochs, event_ids, tmin, tmax, color_dict, picks, title, out_dir=None, verbose='INFO', fig_size=(12.8, 7.2), is_plot_timeseries=True, is_plot_topo_map=True):
    mne.set_log_level(verbose=verbose)
    plt.rcParams["figure.figsize"] = fig_size

    if is_plot_timeseries:
        for ch in picks:
            for event_name, event_marker_id in event_ids.items():
                try:
                    y = epochs.crop(tmin, tmax)[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin, tmax, y.shape[-1])
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                                 interpolate=True,
                                 alpha=0.5)
                plt.plot(time_vector, y_mean, c=color_dict[event_name],
                         label='{0}, N={1}'.format(event_name, epochs[event_name].get_data().shape[0]))
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()
            plt.title('{0} - Channel {1}'.format(title, ch))
            if out_dir:
                plt.savefig(os.path.join(out_dir, '{0} - Channel {1}.png'.format(title, ch)))
                plt.clf()
            else:
                plt.show()

    # get the min and max for plotting the topomap
    if is_plot_topo_map:
        evoked = epochs.average()
        vmax_EEG = np.max(evoked.get_data())
        vmin_EEG = np.min(evoked.get_data())

        for event_name, event_marker_id in event_ids.items():
            try:
                epochs[event_name].average().plot_topomap(times=np.linspace(tmin, tmax, 6), size=3., title='{0} {1}'.format(event_name, title), time_unit='s', scalings=dict(eeg=1.), vmax=vmax_EEG, vmin=vmin_EEG)
            except KeyError:  # meaning this event does not exist in these epochs
                continue

# def generate_condition_sequence(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names,
#                                 session_log,
#                                 item_codes,
#                                 srate=200):  # use a fixed sampling rate for the sampling rate to match between recordings
#     # interpolate nan's
#     data_event_marker_array = add_em_ts_to_data(event_markers,
#                                                 event_marker_timestamps,
#                                                 data_array,
#                                                 data_timestamps, session_log,
#                                                 item_codes, srate)
#     block_sequences = extract_block_data(data_event_marker_array, srate)
#     return block_sequences


def flatten_list(l):
    return [item for sublist in l for item in sublist]

def read_file_lines_as_list(path):
    with open(path, 'r') as filehandle:
        out = [line.rstrip() for line in filehandle.readlines()]
    return out

def append_list_lines_to_file(l, path):
    with open(path, 'a') as filehandle:
        filehandle.writelines("%s\n" % x for x in l)

def create_gaze_behavior_events(fixations, saccades, gaze_timestamps, data_timestamps, deviation_threshold=1e-2, null_percentage=0.025, random_seed=42):
    """
    create a new event array that matches the sampling rate of the data timestamps
    the arguements event_timestamps and data_timestamps must be from the same clock
    @rtype: ndarray: the returned event array will be of the same length as the data_timestamps, and the event values are
    synced with the data_timestamps
    """
    _event_array = np.zeros(data_timestamps.shape)
    null_fixation = []
    for f in fixations:
        onset_time = gaze_timestamps[f.onset]
        if onset_time > np.max(data_timestamps):
            break
        if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
            nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
            if f.stim == 'distractor':
                _event_array[nearest_data_index] = 6  # for fixation onset
            elif f.stim == 'target':
                _event_array[nearest_data_index] = 7  # for fixation onset
            elif f.stim == 'novelty':
                _event_array[nearest_data_index] = 8  # for fixation onset
            elif f.stim == 'null':
                null_fixation.append(f)

    random.seed(random_seed)
    for f in random.sample(null_fixation, int(null_percentage * len(null_fixation))):
        onset_time = gaze_timestamps[f.onset]
        if onset_time > np.max(data_timestamps):
            break
        if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
            nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
            _event_array[nearest_data_index] = 9  # for fixation onset

    null_saccades = []
    for s in saccades:
        onset_time = gaze_timestamps[s.onset]
        if onset_time > np.max(data_timestamps):
            break
        if s.from_stim == 'null' and s.to_stim == 'null':
            null_saccades.append(s)
            continue
        if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
            nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
            _event_array[nearest_data_index] = 10  # for saccade onset

    # select a subset of null fixation and null saccade to add
    for s in random.sample(null_saccades, int(null_percentage * len(null_saccades))):
        onset_time = gaze_timestamps[s.onset]
        if onset_time > np.max(data_timestamps):
            break
        if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
            nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
            _event_array[nearest_data_index] = 10  # for saccade onset
    # print('Found gaze behaviors')
    return np.expand_dims(_event_array, axis=0)

def find_fixation_saccade_targets(fixations, saccades, eyetracking_timestamps, data_egm, deviation_threshold=1e-2):
    """
    process a dataset that has gaze behavior marker and gaze marker, use the gaze marker to find
    first identify the gaze marker inside a fixation,
    if there is an gaze marker, find what stimulus type is that gaze marker
        then we set the preceding saccade's destination to the stimulus type
        then we set the following saccade's origin to the stimulus type
    if there is no gaze marker, then this fixation is not on an object

    note the function can run on either exg or eyetracking, we use exg here as it has higher sampling rate and gives
    presumably better synchronization
    @rtype: new lists of fixations and saccades
    """
    gaze_markers = data_egm[-1]
    data_timestamp = data_egm[0]
    fixations_new = []
    for f in fixations:
        # onset = f.preceding_saccade.onset
        onset_time = eyetracking_timestamps[f.onset]
        offset_time = eyetracking_timestamps[f.offset]
        stim = 'null'
        if np.abs(data_timestamp - onset_time).min() < deviation_threshold:
            data_onset = np.abs(data_timestamp - onset_time).argmin()
            data_offset = np.abs(data_timestamp - offset_time).argmin()
            gm = gaze_markers[data_onset:data_offset]
            unique_markers = [gm[i] for i in sorted(np.unique(gm, return_index=True)[1])]  # a sorted unique list
            if len(unique_markers) == 2 and np.max(unique_markers) == 1:
                stim = 'distractor'
            elif len(unique_markers) == 2 and np.max(unique_markers) == 2:
                stim = 'target'
            elif len(unique_markers) == 2 and np.max(unique_markers) == 3:
                stim = 'novelty'
            elif len(unique_markers) > 2:  # TODO
                stim = 'mixed'
            temp = copy(f)
            temp.preceding_saccade.to_stim = stim
            temp.following_saccade.from_stim = stim
            temp.stim = stim
            fixations_new.append(temp)

    len([f for f in fixations_new if f.stim == 'target']) / len(fixations_new)
    return fixations_new

