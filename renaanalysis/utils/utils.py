import os
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from mne import Epochs
from scipy import signal
from scipy.signal import spectrogram

from renaanalysis.eye.eyetracking import Saccade, GazeRayIntersect
from renaanalysis.params.params import conditions, item_marker_names, \
    tmin_pupil_viz, tmax_pupil_viz, eeg_picks, tmin_eeg_viz, tmax_eeg_viz, eeg_channel_names, \
    ecg_ch_name, eeg_montage, eventmarker_chs, \
    proxy_eog_ch_names, random_seed
from renaanalysis.utils.Event import Event, get_closest_event_attribute_before, get_indices_from_transfer_timestamps, \
    add_event_meta_info, \
    get_block_startend_times, get_last_block_end_time


def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item


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

def get_practice_block_marker(block_marker, practice_length=5):
    rtn = np.array([False] * len(block_marker))
    rtn[:practice_length] = True
    return rtn

def get_block_events(event_markers, event_marker_timestamps):
    events = []
    block_ids = event_markers[eventmarker_chs.index('BlockIDStartEnd'), :]
    block_id_timestamps = event_marker_timestamps[block_ids != 0]
    block_ids = block_ids[block_ids != 0]

    block_condition = event_markers[eventmarker_chs.index('BlockMarker'), :]
    block_conditions_timestamps = event_marker_timestamps[np.logical_and(block_condition != 0, [bm in conditions.values() for bm in block_condition])]
    block_condition = block_condition[np.logical_and(block_condition != 0, [bm in conditions.values() for bm in block_condition])]  # check both non-zero (there is an event), and the marker is for a condition, i.e., not for change of metablocks

    # check if the last block is complete
    if block_ids[-1] > 0:
        block_ids = block_ids[:-1]
        block_id_timestamps = block_id_timestamps[:-1]
        block_condition = block_condition[:-1]
        block_conditions_timestamps = block_conditions_timestamps[:-1]

    assert np.all(block_id_timestamps[::2] == block_conditions_timestamps)  # check the timestamps for the conditionBlockID matches that of conditionBlock conditions

    block_is_practice = get_practice_block_marker(block_condition)
    # add the block starts
    for b_id, b_timestamp, b_condition, b_is_practice in zip(block_ids[::2], block_id_timestamps[::2], block_condition, block_is_practice):
        events.append(Event(b_timestamp, block_id=abs(b_id), block_condition=b_condition, is_block_start=True, block_is_practice=b_is_practice))

    for b_id, b_timestamp, b_condition, b_is_practice in zip(block_ids[1::2], block_id_timestamps[1::2], block_condition, block_is_practice):
        events.append(Event(b_timestamp, block_id=abs(b_id), block_condition=b_condition, is_block_end=True, block_is_practice=b_is_practice))

    # add the meta block events
    meta_block_marker = event_markers[eventmarker_chs.index('BlockMarker'), :]
    meta_block_indices = np.logical_and(meta_block_marker != 0, [bm not in conditions.values() for bm in meta_block_marker])
    meta_block_marker_timestamps = event_marker_timestamps[meta_block_indices]
    meta_block_marker = meta_block_marker[meta_block_indices]

    for m_marker, b_timestamp in zip(meta_block_marker, meta_block_marker_timestamps):
        events.append(Event(b_timestamp, meta_block=m_marker))

    return events

def get_dtn_events(event_markers, event_marker_timestamps, block_events):
    """
    this function must be called after processing the block events so the block information can
    be added to ecah of the dtn events
    :param event_markers:
    :param event_marker_timestamps:
    :param block_events:
    :return:
    """
    events = []

    dtn = event_markers[eventmarker_chs.index('DTN'), :]
    mask = np.logical_and(event_marker_timestamps < get_last_block_end_time(block_events), dtn != 0)

    dtn_timestamps = event_marker_timestamps[mask]

    item_ids = event_markers[eventmarker_chs.index('itemID'), mask]
    obj_dists = event_markers[eventmarker_chs.index('objDistFromPlayer'), mask]
    carousel_speed = event_markers[eventmarker_chs.index('CarouselSpeed'), mask]
    carousel_angle = event_markers[eventmarker_chs.index('CarouselAngle'), mask]
    ts_hand_left = event_markers[eventmarker_chs.index('TSHandLeft'), mask]
    ts_hand_right = event_markers[eventmarker_chs.index('TSHandRight'), mask]
    dtn = dtn[dtn != 0]

    for i, dtn_time in enumerate(dtn_timestamps):
        e = Event(dtn_time, dtn=abs(dtn[i]), item_id=item_ids[i], obj_dist=obj_dists[i])
        e = add_event_meta_info(e, block_events)
        if e.block_condition == conditions['Carousel']:
            e.carousel_speed, e.carousel_angle = carousel_speed[i], carousel_angle[i]
        # if e.block_condition == conditions['TS']  # TODO add ts events

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


def get_gaze_ray_events(item_markers, item_marker_timestamps, events, long_gaze_threshold=0.15):
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

        for item_index, (i_b_gr, i_b_iid, i_b_dtn, i_b_obj_dist) in enumerate(zip(b_gazeray, b_itemids, b_dtns, b_obj_dist)):
            if np.any(i_b_gr != 0):  # if gaze happen on this object
                gaze_ray_diff = np.diff(np.concatenate([[0.], i_b_gr]))
                gaze_ray_onset_times = b_item_timestamps[gaze_ray_diff == 1]
                gaze_ray_offset_times = b_item_timestamps[gaze_ray_diff == -1]

                if len(gaze_ray_onset_times) > len(gaze_ray_offset_times):  # if the last gaze ray started without offset, then put the offset at the end of the block
                    if (len(gaze_ray_onset_times) == 1 and len(gaze_ray_offset_times) == 0) or gaze_ray_onset_times[-1] > gaze_ray_offset_times[-1]:
                        print("Gaze ray onset without offset")
                        gaze_ray_offset_times = np.concatenate([gaze_ray_offset_times, [b_item_timestamps[-1]]])

                gaze_ray_onset_item_ids = i_b_iid[gaze_ray_diff == 1]
                item_dtns = np.unique(i_b_dtn[i_b_dtn!=0])
                assert len(item_dtns) == 1
                gaze_ray_onset_dist = i_b_obj_dist[gaze_ray_diff == 1]

                found_first_long_gaze = False
                for onset_time, offset_time, item_id, item_distance in zip(gaze_ray_onset_times, gaze_ray_offset_times, gaze_ray_onset_item_ids, gaze_ray_onset_dist):
                    if not found_first_long_gaze and (offset_time - onset_time > long_gaze_threshold):
                        found_first_long_gaze = True
                        is_first_long_gaze = True
                    else:
                        is_first_long_gaze = False
                    e = GazeRayIntersect(onset_time, onset_time, offset_time, block_condition=block_conditions[j], block_id=block_ids[j], dtn=item_dtns[0], item_index=item_index, item_id=item_id, obj_dist=item_distance, is_first_long_gaze=is_first_long_gaze)
                    if block_conditions[j] == conditions['Carousel']:
                        e.carousel_speed = get_closest_event_attribute_before(events, onset_time, 'carousel_speed', lambda x: x.dtn_onffset)
                        e.carousel_angle = get_closest_event_attribute_before(events, onset_time, 'carousel_angle', lambda x: x.dtn_onffset)
                    rtn.append(e)
    return rtn


def generate_pupil_event_epochs(data_, data_channels, data_channel_types, event_ids, tmin_pupil=-1., tmax_pupil=3., resample_rate=None, srate=200, verbose='WARNING', n_jobs=1):  # use a fixed sampling rate for the sampling rate to match between recordings
    mne.set_log_level(verbose=verbose)

    info = mne.create_info(
        data_channels,
        sfreq=srate,
        ch_types=data_channel_types)
    raw = mne.io.RawArray(data_.transpose(), info)

    found_events = mne.find_events(raw, stim_channel='stim', shortest_event=1)

    # pupil epochs
    epochs_pupil = Epochs(raw, events=found_events, event_id=event_ids,
                          tmin=tmin_pupil,
                          tmax=tmax_pupil,
                          baseline=(-0.5, 0.0),
                          preload=True,
                          verbose=False,
                          picks=['pupil_left', 'pupil_right'])
    if resample_rate is not None:
        epochs_pupil = epochs_pupil.resample(resample_rate, n_jobs=n_jobs)

    if len(epochs_pupil) < len(found_events):
        warnings.warn(f"generate_pupil_event_epochs: generated fewer than found_event number of epochs, possibly due to incomplete last epoch. Found {len(found_events)} events. But have {len(epochs_pupil)} epochs.")
    return epochs_pupil, epochs_pupil.events[:, 2]


def rescale_merge_exg(data_array_EEG, data_array_ECG, data_array_EOG):
    data_array_EEG = data_array_EEG * 1e-6
    data_array_ECG = data_array_ECG * 1e-6
    data_array_EOG = data_array_EOG * 1e-6

    data_array_ECG = (data_array_ECG[0] - data_array_ECG[1])[None, :]
    data_array = np.concatenate([data_array_EEG, data_array_ECG, data_array_EOG])
    return data_array

def generate_eeg_event_epochs(raw, event_ids, tmin, tmax, resample_rate=None, include_last=False):
    found_events = mne.find_events(raw, stim_channel='stim', shortest_event=1)

    # event_durations = []
    # event_indices = np.argwhere(a != 0)[:, 1]
    # for i, event_index in enumerate(event_indices):
    #     event_durations.append(event_index - event_indices[i-1])
    # event_durations = np.array(event_durations)
    if not include_last:
        tmax -= 1/raw.info['sfreq']

    epochs = Epochs(raw, events=found_events, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=(-0.1, 0.0),preload=True,verbose=False, picks='eeg')
    if resample_rate is not None and epochs.info['sfreq'] != resample_rate:
        epochs = epochs.resample(resample_rate)
    return epochs, epochs.events[:, 2]


def visualize_pupil_epochs(epochs, event_ids, colors, title='', srate=200, verbose='INFO', fig_size=(25.6, 14.4), show=True):
    if colors is None:
        colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}
    plt.rcParams["figure.figsize"] = fig_size
    mne.set_log_level(verbose=verbose)
    # epochs = epochs.apply_baseline((0.0, 0.0))
    for e_name, e_id in event_ids.items():
        y = epochs[e_name].get_data()
        assert np.sum(np.isnan(y)) == 0
        if len(y) == 0:
            print("visualize_pupil_epochs: all epochs bad, skipping")
            continue
        y = np.mean(y, axis=1)  # average left and right
        y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')
        y_mean = np.mean(y, axis=0)
        # y_mean = y_mean - y_mean[int(abs(tmin_pupil_viz) * srate)]  # baseline correct
        y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
        y2 = y_mean - scipy.stats.sem(y, axis=0)  # this is the lower envelope

        time_vector = np.linspace(tmin_pupil_viz, tmax_pupil_viz, y.shape[-1])
        if not (np.any(np.isnan(y1)) or np.any(np.isnan(y2))):
            plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=colors[e_name],
                             interpolate=True,
                             alpha=0.5)
        plt.plot(time_vector, y_mean, c=colors[e_name],
                 label='{0}, N={1}'.format(e_name, y.shape[0]))
    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
    plt.legend()

    if show:
        plt.title(title)
        plt.show()


# TODO visualize_eeg_epochs
def visualize_eeg_epochs(epochs, event_groups, colors, title='', tmin_eeg_viz=-0.1, tmax_eeg_viz=0.8, low_frequency=1, high_frequency=40, eeg_picks=None, out_dir=None, verbose='INFO', fig_size=(12.8, 7.2), is_plot_PSD=True, is_plot_timeseries=True, is_plot_topo_map=False, gaze_behavior=None, is_plot_ERD=True, sfreq=200):
    if colors is None:
        colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}
    mne.set_log_level(verbose=verbose)
    plt.rcParams["figure.figsize"] = fig_size

    if is_plot_timeseries:
        for ch in eeg_picks:
            for event_name, events in event_groups.items():
                try:
                    y = epochs.crop(tmin_eeg_viz, np.min([tmax_eeg_viz, epochs.tmax]))[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin_eeg_viz, tmax_eeg_viz, y.shape[-1])
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=colors[event_name], interpolate=True, alpha=0.5)
                plt.plot(time_vector, y_mean, c=colors[event_name], label='{0}, N={1}'.format(event_name, y.shape[0]))
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
    if is_plot_PSD:
        nperseg = 256
        low_frequency = 1
        high_frequency = 30
        for event_name, events in event_groups.items():
            for ch in eeg_picks:
                if ch == 'C3' or ch == 'C4':
                    try:
                        y = epochs.crop(tmin_eeg_viz, np.min([tmax_eeg_viz, epochs.tmax]))[event_name].pick_channels([ch]).get_data().squeeze(1)
                    except KeyError:  # meaning this event does not exist in these epochs
                        continue
                    n_signals = len(y)
                    # Compute and average the PSDs
                    avg_psd = np.zeros(nperseg // 2 + 1)
                    for sig in y:
                        freq, psd = signal.welch(sig, fs=sfreq, nperseg=nperseg)
                        avg_psd += psd
                    avg_psd /= n_signals

                    # Plot the average PSD
                    min_freq_index = np.argmin(np.abs(freq - low_frequency))
                    max_freq_index = np.argmin(np.abs(freq - high_frequency))
                    plot_freq = freq[min_freq_index:max_freq_index]
                    plt.plot(plot_freq, np.log10(avg_psd[min_freq_index:max_freq_index]), label=ch)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Power/Frequency')
                    plt.title(f'Average Power Spectral Density of class {event_name}')
                    plt.legend()
                    plt.grid()
            plt.show()

    if is_plot_ERD:
        nperseg = 128  # Number of data points per segment
        noverlap = nperseg // 2  # Overlap between segments
        nfft = 512  # Number of FFT points

        n_timepoints = 1000
        freq_min = 1
        freq_max = 30
        for ch in eeg_picks:
            if ch == 'C3' or ch == 'C4':
                for event_name, events in event_groups.items():
                    try:
                        y = epochs.crop(tmin_eeg_viz, np.min([tmax_eeg_viz, epochs.tmax]))[event_name].pick_channels([ch]).get_data().squeeze(1)
                    except KeyError:  # meaning this event does not exist in these epochs
                        continue
                    spectrograms = []
                    frequencies = []
                    for i in range(len(y)):
                        f, t, Sxx = spectrogram(y[i], fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                                nfft=nfft)
                        spectrograms.append(Sxx)
                        frequencies.append(f)
                    mean_spectrograms = np.mean(np.array(spectrograms), axis=0)
                    freqs = np.array(frequencies)[0]
                    min_freq_index = np.argmin(np.abs(freqs - freq_min))
                    max_freq_index = np.argmin(np.abs(freqs - freq_max))
                    im = plt.imshow(np.log10(mean_spectrograms[min_freq_index:max_freq_index]), cmap='jet', aspect='auto',
                               extent=[t.min(), t.max(), freqs[min_freq_index], freqs[max_freq_index]])
                    plt.colorbar(im)
                    plt.title(f'Channel {ch}, class {event_name}')
                    plt.ylabel('Frequency (Hz)')
                    plt.xlabel('Time (s)')
                    plt.tight_layout()
                    plt.show()

    # get the min and max for plotting the topomap
    if is_plot_topo_map:
        evoked = epochs.average()
        vmax_EEG = np.max(evoked.get_data())
        vmin_EEG = np.min(evoked.get_data())

        for event_name, events in event_groups.items():
            try:
                try:
                    epochs[event_name].average().plot_topomap(times=np.linspace(tmin_eeg_viz, tmax_eeg_viz, 6), size=3., title='{0} {1}'.format(event_name, title), time_unit='s', scalings=dict(eeg=1.), vlim=(vmin_EEG, vmax_EEG))
                except TypeError:
                    epochs[event_name].average().plot_topomap(times=np.linspace(tmin_eeg_viz, tmax_eeg_viz, 6), size=3., time_unit='s', scalings=dict(eeg=1.), vlim=(vmin_EEG, vmax_EEG))
            except KeyError:  # meaning this event does not exist in these epochs
                continue

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def append_list_lines_to_file(l, path):
    with open(path, 'a') as filehandle:
        filehandle.writelines("%s\n" % x for x in l)

def preprocess_standard_eeg(raw, ica_path, montage=mne.channels.make_standard_montage('biosemi64'),
                            resample_rate = None,
                            lowcut_eeg=1, lowcut_ecg='0.67',
                            lowcut_eog=0.3, highcut_eeg=50.,
                            highcut_ecg=40., highcut_eog=35, bad_channels=None,
                            is_running_ica=True, is_regenerate_ica=True, is_ica_selection_inclusive=True,
                            ocular_artifact_mode='proxy', blink_ica_threshold=np.linspace(3., 2., 5), eyemovement_ica_threshold=np.linspace(2.5, 1.5, 5),
                            n_jobs=1):
    ch_names = montage.ch_names
    eeg_data = raw.get_data(picks='eeg')
    srate = raw.info['sfreq']
    proxy_horizontal_eog_data = eeg_data[(ch_names.index('F7'), ch_names.index('F8')), :] - eeg_data[ch_names.index('Fpz'), :]
    exg_data = np.concatenate([eeg_data, proxy_horizontal_eog_data])

    data_channels = ch_names + proxy_eog_ch_names
    data_channel_types = ['eeg'] * len(eeg_data) + ['eog'] * 2
    info = mne.create_info(data_channels, sfreq=srate, ch_types=data_channel_types)  # with 3 additional info markers and design matrix
    annotations = raw.annotations

    raw = mne.io.RawArray(exg_data, info)
    raw.set_annotations(annotations)
    raw.set_montage(montage)

    raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)

    if bad_channels is not None: # TODO data pipeline does not give bad channels to this function right now
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(method={'eeg': 'spline'}, verbose='INFO')

    raw = raw.filter(l_freq=lowcut_eeg, h_freq=highcut_eeg, n_jobs=n_jobs, picks='eeg')  # bandpass filter for brain
    # raw = raw.filter(l_freq=lowcut_ecg, h_freq=highcut_ecg, n_jobs=n_jobs, picks='ecg')  # bandpass filter for heart
    raw = raw.filter(l_freq=lowcut_eog, h_freq=highcut_eog, n_jobs=n_jobs, picks='eog')  # bandpass filter for eye
    notch_freqs = [x for x in np.arange(60, 241, 60) if x < srate / 2.]
    raw = raw.notch_filter(freqs=notch_freqs, filter_length='auto', n_jobs=n_jobs)

    if resample_rate is not None:
        raw = raw.resample(resample_rate, n_jobs=n_jobs)
    if is_running_ica:
        if is_regenerate_ica or (not os.path.exists(ica_path + '.txt') or not os.path.exists(ica_path + '-ica.fif')):
            ica = mne.preprocessing.ICA(n_components=20, random_state=random_seed, max_iter=800)
            ica.fit(raw, picks='eeg')

            # if 'ECG00' in raw.ch_names:
            #     ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG00', method='correlation', threshold='auto')
            #     # ica.plot_scores(ecg_scores)
            #     if len(ecg_indices) > 0:
            #         [print('Found ECG component at ICA index {0} with score {1}, adding to ICA exclude'.format(x, ecg_scores[x])) for x in ecg_indices]
            #         ica.exclude += ecg_indices
            #     else:
            #         print('No channel found to be significantly correlated with ECG, skipping auto ECG artifact removal')
            # else:
            #     print('No ECG channel found, skipping auto ECG artifact removal')
            if ocular_artifact_mode == 'proxy':
                print("Proxying blink with Fpz, and left right eye movements with F8-Fpz, F7-Fpz")

                blink_indices = []
                for z_score_threshold in blink_ica_threshold:
                    blink_indices, blink_scores = ica.find_bads_eog(raw, ch_name='Fpz', threshold=z_score_threshold)
                    if len(blink_indices) > 0:
                        [print(f'With z threshold {z_score_threshold}, found Blink component at ICA index {x} with score {blink_scores[x]}, adding to ICA exclude') for x in blink_indices]
                        ica.exclude += blink_indices
                        break
                if len(blink_indices) == 0:
                    warnings.warn('HIGHLY UNLIKELY TO HAPPEN: No channel found to be significantly correlated with blink, skipping auto blink artifact removal')

                eyemovement_indices = []
                for z_score_threshold in eyemovement_ica_threshold:
                    eyemovement_indices, eyemovement_scores = ica.find_bads_eog(raw, ch_name=proxy_eog_ch_names, threshold=z_score_threshold)
                    if len(eyemovement_indices) > 0:
                        [print(f'Found Eye Movement component at ICA index {x} with score: [left {eyemovement_scores[0][x]}] [right {eyemovement_scores[1][x]}], adding to ICA exclude') for x in eyemovement_indices]
                        ica.exclude += eyemovement_indices
                        break
                if len(eyemovement_indices) == 0:
                    warnings.warn('HIGHLY UNLIKELY TO HAPPEN: No channel found to be significantly correlated with Horizontal Eyemovement, skipping auto eyemovement artifact removal')

                ica.plot_sources(raw)
                ica.plot_components()
            elif ocular_artifact_mode == 'manual':
                ica.plot_sources(raw)
                ica.plot_components()
                if is_ica_selection_inclusive:
                    ica_excludes = input("Enter manual ICA components to exclude (use space to deliminate): ")
                    if len(ica_excludes) > 0: ica.exclude += [int(x) for x in ica_excludes.split(' ') if x.isdigit()]
                else:
                    ica_includes = input("Enter manual ICA components to INCLUDE (use space to deliminate): ")
                    ica_includes = [int(x) for x in ica_includes.split(' ')]
                    if len(ica_includes) > 0: ica.exclude += [int(x) for x in range(ica.n_components) if x not in ica_includes]
                    print('Excluding ' + str([int(x) for x in range(ica.n_components) if x not in ica_includes]))
            else:
                raise ValueError('Invalid ocular_artifact_mode: ' + ocular_artifact_mode)
            if ica_path is not None:
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
        ica.apply(raw)

    return raw


def preprocess_session_eeg(data, timestamps, ica_path, srate=2048, resample_rate=None, lowcut_eeg=1, lowcut_ecg='0.67', lowcut_eog=0.3, highcut_eeg=50., highcut_ecg=40., highcut_eog=35, bad_channels=None, is_running_ica=True, is_regenerate_ica=True, is_ica_selection_inclusive=True, ocular_artifact_mode='proxy', n_jobs=20):
    """

    :param data:
    :param timestamps:
    :param ica_path:
    :param srate:
    :param lowcut_eeg:
    :param highcut_eeg:
    :param bad_channels:
    :param is_running_ica:
    :param is_ica_selection_inclusive:
    :param ocular_artifact_mode: can be 'proxy' or manual
    :param n_jobs:
    :return:
    """
    eeg_data = data[0][1:65, :]  # take only the EEG channels
    ecg_data = data[0][65:67, :]
    proxy_horizontal_eog_data = eeg_data[(eeg_channel_names.index('F7'), eeg_channel_names.index('F8')), :] - eeg_data[eeg_channel_names.index('Fpz'), :]

    exg_data = rescale_merge_exg(eeg_data, ecg_data, proxy_horizontal_eog_data)
    # eog data is proxied

    data_channels = ['timestamps'] + eeg_channel_names + [ecg_ch_name] + proxy_eog_ch_names + ['stim']
    data_channel_types = ['misc'] + ['eeg'] * len(eeg_channel_names) + ['ecg'] + ['eog'] * 2 + ['stim']
    info = mne.create_info(data_channels, sfreq=srate, ch_types=data_channel_types)  # with 3 additional info markers and design matrix
    raw = mne.io.RawArray(np.concatenate([timestamps[None, :], exg_data, np.zeros([1, len(timestamps)])], axis=0), info)
    raw.set_montage(eeg_montage)
    raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)

    if bad_channels is not None: # TODO data pipeline does not give bad channels to this function right now
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(method={'eeg': 'spline'}, verbose='INFO')

    raw = raw.filter(l_freq=lowcut_eeg, h_freq=highcut_eeg, n_jobs=n_jobs, picks='eeg')  # bandpass filter for brain
    raw = raw.filter(l_freq=lowcut_ecg, h_freq=highcut_ecg, n_jobs=n_jobs, picks='ecg')  # bandpass filter for heart
    raw = raw.filter(l_freq=lowcut_eog, h_freq=highcut_eog, n_jobs=n_jobs, picks='eog')  # bandpass filter for eye

    notch_freqs = [x for x in np.arange(60, 241, 60) if x < srate / 2.]
    raw = raw.notch_filter(freqs=notch_freqs, filter_length='auto', n_jobs=n_jobs)
    if resample_rate is not None:
        raw = raw.resample(resample_rate, n_jobs=n_jobs)

    if is_running_ica:
        if is_regenerate_ica or (not os.path.exists(ica_path + '.txt') or not os.path.exists(ica_path + '-ica.fif')):
            ica = mne.preprocessing.ICA(n_components=20, random_state=random_seed, max_iter=800)
            ica.fit(raw, picks='eeg')
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG00', method='correlation', threshold='auto')
            # ica.plot_scores(ecg_scores)
            if len(ecg_indices) > 0:
                [print('Found ECG component at ICA index {0} with score {1}, adding to ICA exclude'.format(x, ecg_scores[x])) for x in ecg_indices]
                ica.exclude += ecg_indices
            else:
                print('No channel found to be significantly correlated with ECG, skipping auto ECG artifact removal')

            if ocular_artifact_mode == 'proxy':
                print("Proxying blink with Fpz, and left right eye movements with F8-Fpz, F7-Fpz")

                blink_indices = []
                for z_score_threshold in np.linspace(3., 2., 5):
                    blink_indices, blink_scores = ica.find_bads_eog(raw, ch_name='Fpz', threshold=z_score_threshold)
                    if len(blink_indices) > 0:
                        [print(f'With z threshold {z_score_threshold}, found Blink component at ICA index {x} with score {blink_scores[x]}, adding to ICA exclude') for x in blink_indices]
                        ica.exclude += blink_indices
                        break
                if len(blink_indices) == 0:
                    warnings.warn('HIGHLY UNLIKELY TO HAPPEN: No channel found to be significantly correlated with blink, skipping auto blink artifact removal')

                eyemovement_indices = []
                for z_score_threshold in np.linspace(2.5, 1.5, 5):
                    eyemovement_indices, eyemovement_scores = ica.find_bads_eog(raw, ch_name=proxy_eog_ch_names, threshold=z_score_threshold)
                    if len(eyemovement_indices) > 0:
                        [print(f'Found Eye Movement component at ICA index {x} with score: [left {eyemovement_scores[0][x]}] [right {eyemovement_scores[1][x]}], adding to ICA exclude') for x in eyemovement_indices]
                        ica.exclude += eyemovement_indices
                        break
                if len(eyemovement_indices) == 0:
                    warnings.warn('HIGHLY UNLIKELY TO HAPPEN: No channel found to be significantly correlated with Horizontal Eyemovement, skipping auto eyemovement artifact removal')

                ica.plot_sources(raw)
                ica.plot_components()
            elif ocular_artifact_mode == 'manual':
                ica.plot_sources(raw)
                ica.plot_components()
                if is_ica_selection_inclusive:
                    ica_excludes = input("Enter manual ICA components to exclude (use space to deliminate): ")
                    if len(ica_excludes) > 0: ica.exclude += [int(x) for x in ica_excludes.split(' ') if x.isdigit()]
                else:
                    ica_includes = input("Enter manual ICA components to INCLUDE (use space to deliminate): ")
                    ica_includes = [int(x) for x in ica_includes.split(' ')]
                    if len(ica_includes) > 0: ica.exclude += [int(x) for x in range(ica.n_components) if x not in ica_includes]
                    print('Excluding ' + str([int(x) for x in range(ica.n_components) if x not in ica_includes]))

            if ica_path is not None:
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
        ica.apply(raw)
    # raw.plot(scalings='auto')
    # reconst_raw.plot(show_scrollbars=False, scalings='auto')

    return raw, raw.get_data(picks='timestamps')[0]  # return includes the timestamps squeezing the first dimension

def validate_get_epoch_args(event_names, event_filters):
    try:
        assert len(event_filters) == len(event_names)
    except AssertionError:
        raise ValueError('Number of event names must match the number of event filters')


def remove_value(lst, val):
    return list(filter(lambda x: x != val, lst))


def add_annotations_to_raw(raw, new_annotations):
    """
    Add new annotations to an MNE Raw object.

    Parameters:
        raw (mne.io.Raw): The MNE Raw object with existing annotations.
        new_annotations (mne.Annotations): The new annotations to add.

    Returns:
        mne.io.Raw: The MNE Raw object with both existing and new annotations.
    """
    old_annotations = raw.annotations

    # Combine the existing and new annotations using set operations
    all_onsets = list(old_annotations.onset) + list(new_annotations.onset)
    all_durations = list(old_annotations.duration) + list(new_annotations.duration)
    all_descriptions = list(old_annotations.description) + list(new_annotations.description)

    # Create the new combined annotations
    all_annotations = mne.Annotations(onset=all_onsets, duration=all_durations, description=all_descriptions)

    # Set the new annotations to the raw data
    raw.set_annotations(all_annotations)

    return raw


def is_button_after_oddball(oddball_time, button_times, response_threshold):

    after_oddball_button_times = button_times[button_times > oddball_time] - oddball_time
    return np.any(after_oddball_button_times > 0) and np.any(after_oddball_button_times < response_threshold)

