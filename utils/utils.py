import os
import random
from copy import copy

import scipy
from scipy.interpolate import interp1d

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

from eye.eyetracking import Saccade
from params import *
from utils.Event import Event, get_closest_event_attribute_before, get_indices_from_transfer_timestamps, add_event_meta_info, \
    get_block_startend_times


def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item

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

def get_practice_block_marker(block_marker, practice_length=4):
    rtn = np.array([False] * len(block_marker))
    rtn[:practice_length] = True
    return rtn

def get_block_events(event_markers, event_marker_timestamps):
    events = []
    block_ids = event_markers[eventmarker_preset["ChannelNames"].index('BlockIDStartEnd'), :]
    block_id_timestamps = event_marker_timestamps[block_ids != 0]
    block_ids = block_ids[block_ids != 0]

    block_condition = event_markers[eventmarker_preset["ChannelNames"].index('BlockMarker'), :]
    assert np.all(block_id_timestamps[::2] == event_marker_timestamps[np.logical_and(block_condition != 0, [bm in conditions.values() for bm in block_condition])])  # check the timestamps for the conditionBlockID matches that of conditionBlock conditions
    block_conditions_timestamps = block_id_timestamps[::2]
    block_condition = block_condition[np.logical_and(block_condition != 0, [bm in conditions.values() for bm in block_condition])]  # check both non-zero (there is an event), and the marker is for a condition, i.e., not for change of metablocks
    block_is_practice = get_practice_block_marker(block_condition)
    # add the block starts
    for b_id, b_timestamp, b_condition, b_is_practice in zip(block_ids[::2], block_id_timestamps[::2], block_condition, block_is_practice):
        events.append(Event(b_timestamp, block_id=abs(b_id), block_condition=b_condition, is_block_start=True, block_is_practice=b_is_practice))

    for b_id, b_timestamp, b_condition, b_is_practice in zip(block_ids[1::2], block_id_timestamps[1::2], block_condition, block_is_practice):
        events.append(Event(b_timestamp, block_id=abs(b_id), block_condition=b_condition, is_block_end=True, block_is_practice=b_is_practice))

    # add the meta block events
    meta_block_marker = event_markers[eventmarker_preset["ChannelNames"].index('BlockMarker'), :]
    meta_block_indices = np.logical_and(meta_block_marker != 0, [bm not in conditions.values() for bm in meta_block_marker])
    meta_block_marker_timestamps = event_marker_timestamps[meta_block_indices]
    meta_block_marker = meta_block_marker[meta_block_indices]

    for m_marker, b_timestamp in zip(meta_block_marker, meta_block_marker_timestamps):
        events.append(Event(b_timestamp, meta_block=m_marker))

    return events

def get_dtn_events(event_markers, event_marker_timestamps, block_events):
    events = []

    dtn = event_markers[eventmarker_preset["ChannelNames"].index('DTN'), :]
    dtn_timestamps = event_marker_timestamps[dtn != 0]
    item_ids = event_markers[eventmarker_preset["ChannelNames"].index('itemID'), dtn != 0]
    obj_dists = event_markers[eventmarker_preset["ChannelNames"].index('objDistFromPlayer'), dtn != 0]
    carousel_speed = event_markers[eventmarker_preset["ChannelNames"].index('CarouselSpeed'), dtn != 0]
    carousel_angle = event_markers[eventmarker_preset["ChannelNames"].index('CarouselAngle'), dtn != 0]
    dtn = dtn[dtn != 0]

    for i, dtn_time in enumerate(dtn_timestamps):
        e = Event(dtn_time, dtn=abs(dtn[i]), item_id=item_ids[i], obj_dist=obj_dists[i])
        e = add_event_meta_info(e, block_events)
        if e.condition == conditions['Carousel']:
            e.carousel_speed, e.carousel_angle = carousel_speed[i], carousel_angle[i]

        e.dtn_onffset = dtn[i] > 0

        events.append(e)
    return events

def get_item_events(event_markers, event_marker_timestamps, item_markers, item_marker_timestamps):
    """
    add LSL timestamps, event markers based on the session log to the data array
    also discard data that falls other side the first and the last block

    add the DTN events for RSVP and carousel, with meta information.
    add the block onsets and offsets, with likert info TODO
    :param event_markers:
    :param event_marker_timestamps:
    :param data_dicts: list of data keys include 'data_array', 'data_timestamps', and 'srate'
    :param session_log:
    :param item_codes:
    :param srate:
    :param pre_first_block_time:
    :param post_final_block_time:
    :return:
    """
    events = []

    events += get_block_events(event_markers, event_marker_timestamps)  # get the block events
    events += get_dtn_events(event_markers, event_marker_timestamps, events)  # dtn events needs the block events to know the conditions
    events += get_gaze_ray_events(item_markers, item_marker_timestamps, events)  # dtn events needs the block events to know the conditions

    # add gaze related events
    return events


def extract_block_data(_data, channel_names, srate, fixations, saccades, pre_block_time=.5, post_block_time=.5):  # event markers is the third last row
    # TODO update this v3 experiment
    block_starts = np.argwhere(_data[channel_names.index('EventMarker'), :] == 4) - int(pre_block_time * srate)# start of a block is denoted by event marker 4
    block_ends = np.argwhere(_data[channel_names.index('EventMarker'), :] == 5) + int(post_block_time * srate)  # end of a block is denoted by event marker 5

    block_sequences = [_data[:, i[0]:j[0]] for i, j in zip(block_starts, block_ends)]
    return block_sequences

    # plt.rcParams["figure.figsize"] = (60, 15)
    # b = block_sequences[0]
    # b = np.copy(b)
    # t = b[0]
    # p = b[channel_names.index('left_pupil_size'), :]
    # p[p == 0] = np.nan
    # p = interpolate_nan(p)
    # p = p * 1e2
    # xy = b[[channel_names.index('gaze_forward_x'), channel_names.index('gaze_forward_y')], :]
    # xy_deg = (180 / math.pi) * np.arcsin(xy)
    # dxy = np.diff(xy_deg, axis=1, prepend=xy_deg[:, :1])
    # dtheta = np.linalg.norm(dxy, axis=0)
    # velocities = dtheta / np.diff(t, prepend=1)
    # velocities[0] = 0.
    # ################
    # plt.plot(t, p)
    # for f in [f for f in fixations if f.onset_time > t.min() and f.offset_time < t.max()]:
    #     plt.axvspan(f.onset_time, f.offset_time, alpha=0.5, color=event_color_dict[f.stim])
    # for s in [s for s in saccades if s.onset_time > t.min() and s.offset_time < t.max()]:
    #     plt.axvspan(s.onset_time, s.offset_time, alpha=0.5, color=event_color_dict['saccade'])
    #
    # for i, e in enumerate(b[channel_names.index('EventMarker'), :]):
    #     if e != 0:
    #         plt.scatter(t[i], p[i] + 0.1, alpha = 0.5, color=event_marker_color_dict[e], marker='v', s=200)
    #
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Pupil Size (mm)')
    # plt.title('Block sequence with gaze behavior color bars and event markers \n Condition RSVP, Participant 1, Session 1')
    # plt.show()
    # ###################
    # plt.plot(t, p)
    # for i, e in enumerate(b[channel_names.index('EventMarker'), :]):
    #     if e != 0:
    #         plt.axvline(t[i], alpha = 0.5, linewidth=2, color=em_color_code_dict[e])
    #
    # for i, e in enumerate(b[channel_names.index('GazeMarker'), :]):
    #     if e != 0:
    #         plt.axvline(t[i], linestyle='--', color='orange')
    # plt.xlabel('Time (sec)')
    # plt.title('Gaze ray intersects')
    # plt.show()
    #
    # return block_sequences  # a list of block sequences

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


def get_gaze_ray_events(item_markers, item_marker_timestamps, events):
    """
    the item marker has the gaze events for each object. Its channel represent different objects in each block
    @param item_markers:
    @param item_markers_timestamps:
    @param event_markers:
    @param event_marker_timestamps:
    @param data_array:
    @param session_log:
    @param item_codes:
    @param srate:
    @param verbose:
    @param pre_block_time:
    @param post_block_time:
    @param foveate_value_threshold:
    @param foveate_duration_threshold:
    @return:
    """
    rtn = []

    # get block infos
    block_start_timestamps, block_end_timestamps = get_block_startend_times(events)
    block_conditions = [e.block_condition for e in events if e.is_block_start]
    block_ids = [e.block_id for e in events if e.is_block_start]

    item_block_start_idx = get_indices_from_transfer_timestamps(item_marker_timestamps, block_start_timestamps)
    item_block_end_idx = get_indices_from_transfer_timestamps(item_marker_timestamps, block_end_timestamps)

    block_item_markers = [(item_markers[:, start:end], item_marker_timestamps[start:end])for start, end in zip(item_block_start_idx, item_block_end_idx)]
    for j, (b_item_markers, b_item_timestamps) in enumerate(block_item_markers):
        # if j == 4:
        #     print("this is the end of the practice rounds")

        b_gazeray = b_item_markers[item_marker_names.index('isGazeRayIntersected')::len(item_marker_names), :]  # the gaze ray inter for 30 items in this block
        b_itemids = b_item_markers[item_marker_names.index('itemID')::len(item_marker_names), :]  # the gaze ray inter for 30 items in this block
        b_dtns = b_item_markers[item_marker_names.index('itemDTNType')::len(item_marker_names), :]  # the gaze ray inter for 30 items in this block
        b_obj_dist = b_item_markers[item_marker_names.index('distFromPlayer')::len(item_marker_names), :]  # the gaze ray inter for 30 items in this block

        # b_foveate_angle = b_item_markers[item_marker_names.index('foveateAngle')::len(item_marker_names), :]  # the gaze ray inter for 30 items in this block
        # b_foveate_angle = b_foveate_angle * np.pi / 180

        for i_b_gr, i_b_iid, i_b_dtn, i_b_obj_dist in zip(b_gazeray, b_itemids, b_dtns, b_obj_dist):
            if np.any(i_b_gr != 0):

                ts = b_item_timestamps[i_b_gr != 0][0]  # take the first timestamp where gaze ray intersects

                # if debug and j >= 5:
                #     temp = np.unique(i_b_dtn)[np.unique(i_b_dtn) != 0][0]
                #     print("Gaze ray intersect item DTN is {}, preceding event marker is {}".format(temp, event_marker_dtn))
                #     if temp != event_marker_dtn:
                #         print('hi')

                i_b_obj_dist = i_b_obj_dist[i_b_gr!=0][0]

                i_b_iid = i_b_iid[i_b_gr != 0]
                i_b_dtn = i_b_dtn[i_b_gr != 0]
                assert np.all(i_b_iid == i_b_iid)
                i_b_iid = i_b_iid[0]

                i_b_dtn = np.unique(i_b_dtn)[np.unique(i_b_dtn) != 0][0]

                # i_b_gr = i_b_gr[i_b_gr!=0][0]


                # TODO only keep the first gaze ray event for now, not taking diff because gaze ray intersect is too discrete
                e = Event(ts, gaze_intersect=True, block_condition=block_conditions[j], block_id=block_ids[j], dtn=i_b_dtn, item_id=i_b_iid + 1,obj_dist=i_b_obj_dist)

                if block_conditions[j] == conditions['Carousel']:
                    e.carousel_speed = get_closest_event_attribute_before(events, ts, 'carousel_speed', lambda x: x.dtn_onffset)
                    e.carousel_angle = get_closest_event_attribute_before(events, ts, 'carousel_angle', lambda x: x.dtn_onffset)
                rtn.append(e)
    return rtn


def generate_pupil_event_epochs(data_, data_channels, data_channel_types, tmin, tmax, event_ids_dict, erp_window=(.0, .8), srate=200,
                                verbose='WARNING'):  # use a fixed sampling rate for the sampling rate to match between recordings
    mne.set_log_level(verbose=verbose)

    info = mne.create_info(
        data_channels,
        sfreq=srate,
        ch_types=data_channel_types)  # with 3 additional info markers
    raw = mne.io.RawArray(data_, info)

    # only keep events that are in the block
    epochs_pupil_all = []
    labels_array_all = []
    for stim_channel, event_ids in event_ids_dict.items():
        found_events = mne.find_events(raw, stim_channel=stim_channel)
        unique_events = np.unique(found_events[:, 2])
        events = dict([(event_name, event_code) for event_name, event_code in event_ids.items() if event_code in unique_events])
        # pupil epochs
        if len(events) > 0:
            epochs_pupil = Epochs(raw, events=found_events, event_id=events, tmin=tmin,
                                  tmax=tmax,
                                  baseline=(-0.5, 0.0),
                                  preload=True,
                                  verbose=False, picks=['left_pupil_size', 'right_pupil_size'])
            epochs_pupil_all.append(epochs_pupil)
            labels_array_all.append(epochs_pupil.events[:, 2])
    epochs_pupil_all = mne.concatenate_epochs(epochs_pupil_all)
    labels_array_all = np.concatenate(labels_array_all)
    return epochs_pupil_all, labels_array_all


def rescale_merge_exg(data_array_EEG, data_array_ECG):
    data_array_EEG = data_array_EEG * 1e-6
    data_array_ECG = data_array_ECG * 1e-6
    data_array_ECG = (data_array_ECG[0] - data_array_ECG[1])[None, :]
    data_array = np.concatenate([data_array_EEG, data_array_ECG])
    return data_array

def generate_eeg_event_epochs(data_, data_channels, data_channle_types, ica_path, tmin, tmax, event_ids_dict, erp_window=(.0, .8), srate=2048, verbose='CRITICAL',
                              is_regenerate_ica=False, is_ica_selection_inclusive=True, lowcut=1, highcut=50., resample_srate=128, bad_channels=None):
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
        if is_ica_selection_inclusive:
            ica_excludes = input("Enter manual ICA components to exclude (use space to deliminate): ")
            if len(ica_excludes) > 0: ica.exclude += [int(x) for x in ica_excludes.split(' ')]
        else:
            ica_includes = input("Enter manual ICA components to INCLUDE (use space to deliminate): ")
            ica_includes = [int(x) for x in ica_includes.split(' ')]
            if len(ica_includes) > 0: ica.exclude += [int(x) for x in range(ica.n_components) if x not in ica_includes]
            print('Excluding ' + str([int(x) for x in range(ica.n_components) if x not in ica_includes]))

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

    epochs_all = []
    epochs_ica_cleaned_all = []
    labels_array_all = []
    for stim_channel, event_ids in event_ids_dict.items():
        found_events = mne.find_events(raw, stim_channel=stim_channel)
        unique_events = np.unique(found_events[:, 2])
        events = dict([(event_name, event_code) for event_name, event_code in event_ids.items() if event_code in unique_events])
        if len(events) > 0:
            epochs = Epochs(raw, events=found_events, event_id=events, tmin=tmin,
                            tmax=tmax,
                            baseline=(-0.1, 0.0),
                            preload=True,
                            verbose=False,
                            reject=reject)

            epochs_ICA_cleaned = Epochs(raw_ica_recon, events=found_events, event_id=events,
                                        tmin=tmin,
                                        tmax=tmax,
                                        baseline=(-0.1, 0.0),
                                        preload=True,
                                        verbose=False,
                                        reject=reject)
            epochs_all.append(epochs)
            epochs_ica_cleaned_all.append(epochs_ICA_cleaned)
            labels_array_all.append(epochs.events[:, 2])
    epochs_all = mne.concatenate_epochs(epochs_all)
    epochs_ica_cleaned_all = mne.concatenate_epochs(epochs_ica_cleaned_all)
    labels_array_all = np.concatenate(labels_array_all)
    return epochs_all, epochs_ica_cleaned_all, labels_array_all, raw, raw_ica_recon


def visualize_pupil_epochs(epochs, event_groups, tmin, tmax, title, srate=200, verbose='INFO', fig_size=(25.6, 14.4), gaze_behavior=None):
    plt.rcParams["figure.figsize"] = fig_size
    mne.set_log_level(verbose=verbose)
    # epochs = epochs.apply_baseline((0.0, 0.0))
    for event_name, events in event_groups.items():
        try:
            y = epochs[event_name].get_data()
        except KeyError:  # meaning this event does not exist in these epochs
            continue
        y = interpolate_epoch_zeros(y)  # remove nan
        y = interpolate_epochs_nan(y)  # remove nan
        assert np.sum(np.isnan(y)) == 0
        if len(y) == 0:
            print("visualize_pupil_epochs: all epochs bad, skipping {0}".format(events))
            continue
        y = np.mean(y, axis=1)  # average left and right
        y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

        y_mean = np.mean(y, axis=0)
        y_mean = y_mean - y_mean[int(abs(tmin) * srate)]  # baseline correct
        y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
        y2 = y_mean - scipy.stats.sem(y, axis=0)  # this is the lower envelope

        time_vector = np.linspace(tmin, tmax, y.shape[-1])
        color = color_dict[event_name]
        if not (np.any(np.isnan(y1)) or np.any(np.isnan(y2))):
            plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color,
                             interpolate=True,
                             alpha=0.5)
        plt.plot(time_vector, y_mean, c=color,
                 label='{0}, N={1}'.format(event_name, y.shape[0]))
    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
    plt.legend()

    # plot gaze behavior if any
    if gaze_behavior:
        if type(gaze_behavior[0]) is Saccade:
            durations = [x.duration for x in gaze_behavior if x.epoched]
            plt.twinx()
            n, bins, patches = plt.hist(durations, bins=10)
            plt.ylim(top=max(n) / 0.2)
            plt.ylabel('Saccade duration histogram')

    plt.legend()
    plt.title(title)
    plt.show()


def visualize_eeg_epochs(epochs, event_groups, tmin, tmax, picks, title, out_dir=None, verbose='INFO', fig_size=(12.8, 7.2), is_plot_timeseries=True, is_plot_topo_map=True, gaze_behavior=None):
    mne.set_log_level(verbose=verbose)
    plt.rcParams["figure.figsize"] = fig_size

    if is_plot_timeseries:
        for ch in picks:
            for event_name, events in event_groups.items():
                try:
                    y = epochs.crop(tmin, tmax)[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin, tmax, y.shape[-1])
                color = color_dict[event_name]
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color,
                                 interpolate=True,
                                 alpha=0.5)
                plt.plot(time_vector, y_mean, c=color,
                         label='{0}, N={1}'.format(event_name, y.shape[0]))
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()

            # plot gaze behavior if any
            if gaze_behavior:
                if type(gaze_behavior[0]) is Saccade:
                    durations = [x.duration for x in gaze_behavior if x.epoched]
                    plt.twinx()
                    n, bins, patches = plt.hist(durations, bins=10)
                    plt.ylim(top=max(n) / 0.2)
                    plt.ylabel('Saccade duration histogram')

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

        for event_name, events in event_groups.items():
            try:
                epochs[events].average().plot_topomap(times=np.linspace(tmin, tmax, 6), size=3., title='{0} {1}'.format(event_name, title), time_unit='s', scalings=dict(eeg=1.), vmax=vmax_EEG, vmin=vmin_EEG)
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

# def create_gaze_behavior_events(fixations, saccades, gaze_timestamps, data_timestamps, deviation_threshold=1e-2, null_percentage=0.025, random_seed=42):
#     """
#     create a new event array that matches the sampling rate of the data timestamps
#     the arguements event_timestamps and data_timestamps must be from the same clock
#     @rtype: ndarray: the returned event array will be of the same length as the data_timestamps, and the event values are
#     synced with the data_timestamps
#     """
#     _event_array = np.zeros(data_timestamps.shape)
#     null_fixation = []
#     for f in fixations:
#         onset_time = gaze_timestamps[f.onset]
#         if onset_time > np.max(data_timestamps):
#             break
#         if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
#             nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
#             if f.stim == 'distractor':
#                 _event_array[nearest_data_index] = 6  # for fixation onset on distractor
#                 f.epoched = True
#             elif f.stim == 'target':
#                 _event_array[nearest_data_index] = 7  # for fixation onset on targets
#                 f.epoched = True
#             elif f.stim == 'novelty':
#                 _event_array[nearest_data_index] = 8  # for fixation onset on novelty
#                 f.epoched = True
#             elif f.stim == 'null':  # for fixation onset on nothing
#                 null_fixation.append(f)
#             elif f.stim is None or f.stim == 'mixed':
#                 continue  # ignore fixation with unknown type
#             else:
#                 raise Exception("Unknown fixation to_stim typ: {0}, this should never happen".format(f.stim))
#
#     random.seed(random_seed)
#     for f in random.sample(null_fixation, int(null_percentage * len(null_fixation))):
#         onset_time = gaze_timestamps[f.onset]
#         if onset_time > np.max(data_timestamps):
#             break
#         if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
#             nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
#             _event_array[nearest_data_index] = 9  # for fixation onset
#             f.epoched = True
#
#     null_saccades = []
#     for s in saccades:
#         onset_time = gaze_timestamps[s.onset]
#         if onset_time > np.max(data_timestamps):
#             break
#         if s.from_stim == 'null' or s.to_stim == 'null':
#             null_saccades.append(s)
#             continue
#         if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
#             nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
#
#             if s.to_stim == 'distractor':
#                 _event_array[nearest_data_index] = 10  # for saccade onset to distractor
#                 s.epoched = True
#             elif s.to_stim == 'target':
#                 _event_array[nearest_data_index] = 11  # for saccade onset to targets
#                 s.epoched = True
#             elif s.to_stim == 'novelty':
#                 _event_array[nearest_data_index] = 12  # for saccade onset to novelty
#                 s.epoched = True
#             elif s.to_stim is None or s.to_stim == 'mixed':
#                 continue  # ignore saccade with unknown type
#             else:
#                 raise Exception("Unknown saccade to_stim type {0}, this should never happen".format(s.to_stim))
#
#     # select a subset of null fixation and null saccade to add
#     for s in random.sample(null_saccades, int(null_percentage * len(null_saccades))):
#         onset_time = gaze_timestamps[s.onset]
#         if onset_time > np.max(data_timestamps):
#             break
#         if np.min(np.abs(data_timestamps - onset_time)) < deviation_threshold:
#             nearest_data_index = (np.abs(data_timestamps - onset_time)).argmin()
#             _event_array[nearest_data_index] = 13  # for saccade onset
#             s.epoched = True
#     # print('Found gaze behaviors')
#     return np.expand_dims(_event_array, axis=0)


