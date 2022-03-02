import numpy as np
import scipy
from scipy.interpolate import interp1d
import json

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

from rena.utils.data_utils import RNStream

FIXATION_MINIMAL_TIME = 1e-3 * 141.42135623730952
ITEM_TYPE_ENCODING = {1: 'distractor', 2: 'target', 3: 'novelty'}


def interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
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
    return np.array([[interpolate_nan(x) for x in e] for e in epoch_array])


def interpolate_epoch_zeros(e):
    copy = np.copy(e)
    copy[copy == 0] = np.nan
    return interpolate_epochs_nan(copy)


def add_eventMarkers_LSLTimestamp_to_data(event_markers, event_marker_timestamps, data_array, data_timestamps,
                                          session_log,
                                          item_codes):
    block_num = None
    assert event_markers.shape[0] == 4
    data_event_marker_array = np.zeros(shape=(4, data_array.shape[1]))

    for i in range(event_markers.shape[1]):
        event, info1, info2, info3 = event_markers[:, i]
        data_event_marker_index = (np.abs(data_timestamps - event_marker_timestamps[i])).argmin()

        if str(int(event)) in session_log.keys():  # for start-of-block events
            print('Processing block with ID: {0}'.format(event))
            block_num = event
            data_event_marker_array[0][data_event_marker_index] = 4  # encodes start of a block
            continue
        elif event_markers[0, i - 1] != 0 and event == 0:  # this is the end of a block
            data_event_marker_array[0][data_event_marker_index] = 5  # encodes start of a block
            continue

        if event in item_codes:  # for item events
            targets = session_log[str(int(block_num))]['targets']
            distractors = session_log[str(int(block_num))]['distractors']
            novelties = session_log[str(int(block_num))]['novelties']

            if event in distractors:
                data_event_marker_array[0][data_event_marker_index] = 1
            elif event in targets:
                data_event_marker_array[0][data_event_marker_index] = 2
            elif event in novelties:
                data_event_marker_array[0][data_event_marker_index] = 3
            data_event_marker_array[1:4, data_event_marker_index] = info1, info2, info3
            # print('    Item event {0} is novelty with info {1}'.format(event, str(data_event_marker_array[0:4,
            #                                                                       data_event_marker_index])))

    return np.concatenate([np.expand_dims(data_timestamps, axis=0), data_array, data_event_marker_array], axis=0)
    # return np.concatenate(
    #     [np.expand_dims(data_timestamps, axis=0), data_array[1:, :], data_event_marker_array],
    #     axis=0)  # TODO because LSL cannot stream the raw nano second timestamp, we use the LSL timestamp instead. Convert the timestamps in second to nanosecond to comply with gaze detection code


def add_gaze_event_markers_to_data_array(item_markers, item_markers_timestamps, event_markers, event_marker_timestamps,
                                         data_array, data_timestamps,
                                         session_log, item_codes):
    block_list = []
    assert event_markers.shape[0] == 4
    data_event_marker_array = np.zeros(shape=(4, data_array.shape[1]))

    for i in range(event_markers.shape[1]):
        event, info1, info2, info3 = event_markers[:, i]
        data_event_marker_index = (np.abs(data_timestamps - event_marker_timestamps[i])).argmin()

        if str(int(event)) in session_log.keys():  # for start-of-block events
            print('Processing block with ID: {0}'.format(event))
            block_list.append(event)
            data_event_marker_array[0][data_event_marker_index] = 4  # encodes start of a block
            continue
        elif event_markers[0, i - 1] != 0 and event == 0:  # this is the end of a block
            data_event_marker_array[0][data_event_marker_index] = 5  # encodes start of a block
            continue

    data_block_starts_indices = np.argwhere(
        data_event_marker_array[0, :] == 4)  # start of a block is denoted by event marker 4
    data_block_ends_indices = np.argwhere(
        data_event_marker_array[0, :] == 5)  # end of a block is denoted by event marker 5

    # iterate through blocks
    for block_i, data_start_i, data_end_i in zip(block_list, data_block_starts_indices, data_block_ends_indices):
        targets = session_log[str(int(block_i))]['targets']
        distractors = session_log[str(int(block_i))]['distractors']
        novelties = session_log[str(int(block_i))]['novelties']

        # 1. find the event marker timestamps corresponding to the block start and end
        data_block_start_timestamp = data_timestamps[data_start_i]
        data_block_end_timestamp = data_timestamps[data_end_i]
        # 2. find the nearest timestamp of the block start and end in the item marker timestamps
        item_marker_block_start_index = np.argmin(np.abs(item_markers_timestamps - data_block_start_timestamp))
        item_marker_block_end_index = np.argmin(np.abs(item_markers_timestamps - data_block_end_timestamp))
        item_markers_of_block = item_markers[:, item_marker_block_start_index:item_marker_block_end_index]
        item_markers_timestamps_of_block = item_markers_timestamps[
                                           item_marker_block_start_index:item_marker_block_end_index]

        for i in range(30):
            # this_item_marker = item_markers_of_block[i * 11: (i + 1) * 11, i::30]
            this_item_marker = item_markers_of_block[i: i + 11, i::30]
            this_item_markers_timestamps = item_markers_timestamps_of_block[i::30]
            if len(item_markers_timestamps_of_block) == 0:
                print('Block ID {0} is missing item markers, ignoring this block'.format(block_i))
                continue
            # TODO finish this, now assume every 30
            assert np.all(this_item_marker[1, :] == this_item_marker[
                1, 0])  # verify our assumption that item rotates around every 30 columns  # TODO why did the item marker stopped mid-experiment
            # find if there is gaze ray intersection
            # TODO change this index from 5 to 4 in the future

            gaze_intersect_start_index = np.argwhere(np.diff(this_item_marker[5, :]) == 1)[:, 0]
            gaze_intersect_end_index = np.argwhere(np.diff(this_item_marker[5, :]) == -1)[:, 0]
            if len(gaze_intersect_start_index) > len(
                    gaze_intersect_end_index): gaze_intersect_start_index = gaze_intersect_start_index[
                                                                            :len(gaze_intersect_end_index - 1)]

            # check if the intersects is long enough to warrant a fixation
            gaze_intersected_durations = this_item_markers_timestamps[gaze_intersect_end_index] - \
                                         this_item_markers_timestamps[gaze_intersect_start_index]
            true_fixations_indices = np.argwhere(gaze_intersected_durations > FIXATION_MINIMAL_TIME)[:, 0]
            true_fixation_timestamps = this_item_markers_timestamps[gaze_intersect_start_index[true_fixations_indices]]

            # check if this item is a target/distractor/novelty
            marker_to_insert = 1 if this_item_marker[1, 0] in distractors else 2 if this_item_marker[
                                                                                        1, 0] in targets else 3 if \
                this_item_marker[1, 0] in novelties else -1
            assert marker_to_insert != -1

            # find where in data marker to insert the marker
            data_event_marker_indices = [(np.abs(data_timestamps - x)).argmin() for x in true_fixation_timestamps]
            data_event_marker_array[0][data_event_marker_indices] = marker_to_insert
            if len(true_fixation_timestamps) > 0: print(
                'Found {0} fixations for item {1} of type {2}, in block {3}'.format(len(true_fixation_timestamps),
                                                                                    this_item_marker[1, 0],
                                                                                    ITEM_TYPE_ENCODING[
                                                                                        marker_to_insert], block_i))

        # 3. get the IsGazeRay intersected stream and their timestamps (item marker) keyed by the item count in block

        # 4. for each of the 30 items in the block, find where the IsGazeRay is true
        # 5. insert the gazed event marker in the data_event_marker_array at the data_timestamp nearest to the corresponding item_marker_timestamp

    return np.concatenate([np.expand_dims(data_timestamps, axis=0), data_array, data_event_marker_array], axis=0)


def extract_block_data(data_with_event_marker, data_channel_names, block_end_margin_seconds,
                       srate):  # event markers is the third last row
    # TODO add block end margins and use parameters block_end_margin_seconds
    block_starts = np.argwhere(data_with_event_marker[-4, :] == 4)  # start of a block is denoted by event marker 4
    block_ends = np.argwhere(data_with_event_marker[-4, :] == 5)  # end of a block is denoted by event marker 5
    block_sequences = [data_with_event_marker[:, i[0]:j[0]] for i, j in zip(block_starts, block_ends)]
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


def generate_pupil_event_epochs(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names,
                                session_log,
                                item_codes, tmin, tmax, event_ids, color_dict, title='', is_plotting=True,
                                srate=200):  # use a fixed sampling rate for the sampling rate to match between recordings
    eyetracking_with_event_marker_data = add_eventMarkers_LSLTimestamp_to_data(event_markers,
                                                                               event_marker_timestamps,
                                                                               data_array,
                                                                               data_timestamps, session_log,
                                                                               item_codes)

    info = mne.create_info(
        ['LSLTimestamp'] + data_channel_names + ['EventMarker'] + ["info1", "info2", "info3"],
        sfreq=srate,
        ch_types=['misc'] * (1 + len(data_channel_names)) + ['stim'] + [
            'misc'] * 3)  # with 3 additional info markers
    raw = mne.io.RawArray(eyetracking_with_event_marker_data, info)

    # pupil epochs
    epochs_pupil = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
                          tmax=tmax,
                          baseline=(-0.1, 0.0),
                          preload=True,
                          verbose=False, picks=['left_pupil_size', 'right_pupil_size'])
    # verbose=False, picks=['left_pupil_size', 'right_pupil_size', 'status', 'left_status', 'right_status'])

    # Average epoch data
    if is_plotting:
        for event_name, event_marker_id in event_ids.items():
            if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
                y = epochs_pupil[event_name].get_data()
                y = interpolate_epochs_nan(y)

                time_vector = np.linspace(tmin, tmax, y.shape[-1])

                # y = np.mean(y, axis=1)  # average left and right
                y = y[:, 0, :]  # get the left eye data
                y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

                y = np.array([mne.baseline.rescale(x, time_vector, (-0.1, 0.)) for x in y])
                y1 = np.mean(y, axis=0) + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = np.mean(y, axis=0) - scipy.stats.sem(y, axis=0)  # this is the lower envelope
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                                 interpolate=True,
                                 alpha=0.5)
                plt.plot(time_vector, np.mean(y, axis=0), c=color_dict[event_name],
                         label='{0}, N={1}'.format(event_name, epochs_pupil[event_name].get_data().shape[0]))
        plt.xlabel('Time (sec)')
        plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
        plt.legend()
        plt.title(title)
        plt.show()

        # ERP Image
        for event_name, event_marker_id in event_ids.items():
            if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
                y = epochs_pupil[event_name].get_data()
                y = np.mean(y, axis=1)  # average left and right
                # y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')
                time_vector = np.linspace(tmin, tmax, y.shape[-1])
                plt.imshow(y)
                plt.xticks(np.arange(0, y.shape[1], y.shape[1] / 5),
                           ["{:6.2f}".format(x) for x in np.arange(tmin, tmax, (tmax - tmin) / 5)])
                plt.xlabel('Time (sec)')
                plt.ylabel('Trails')
                plt.legend()
                plt.title('{0}: {1}'.format(title, event_name))
                plt.show()
    # epochs_gaze = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
    #                      tmax=tmax,
    #                      baseline=None,
    #                      preload=True,
    #                      verbose=False, picks=['raw_timestamp', 'status', 'gaze_forward_x', 'gaze_forward_y'])
    labels_array = epochs_pupil.events[:, 2]
    return epochs_pupil, labels_array


def generate_eeg_event_epochs(event_markers, event_marker_timestamps, data_array, data_timestamps,
                              session_log,
                              item_codes, tmin, tmax, event_ids, lowcut=0.5, highcut=50., notch_band_demoninator=200,
                              EEG_fresample=50,
                              srate=2048):  # use a fixed sampling rate for the sampling rate to match between recordings
    # interpolate nan's
    eeg_with_event_marker_data = add_eventMarkers_LSLTimestamp_to_data(event_markers,
                                                                       event_marker_timestamps,
                                                                       data_array,
                                                                       data_timestamps, session_log,
                                                                       item_codes)
    biosemi_64_montage = mne.channels.make_standard_montage('biosemi64')
    data_channel_names = biosemi_64_montage.ch_names
    info = mne.create_info(
        ['LSLTimestamp'] + data_channel_names + ['EventMarker'] + ["info1", "info2", "info3"],
        sfreq=srate,
        ch_types=['misc'] + ['eeg'] * len(data_channel_names) + ['stim'] + [
            'misc'] * 3)  # with 3 additional info markers
    raw = mne.io.RawArray(eeg_with_event_marker_data, info)
    raw.set_montage(biosemi_64_montage)
    raw, _ = mne.set_eeg_reference(raw, 'average',
                                   projection=False)
    raw = raw.filter(l_freq=lowcut, h_freq=highcut)  # bandpass filter
    raw = raw.notch_filter(freqs=np.arange(60, 241, 60), filter_length='auto')

    reject = dict(eeg=600.)
    epochs = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True,
                    verbose=False, reject=reject)
    # verbose=False, picks=['left_pupil_size', 'right_pupil_size', 'status', 'left_status', 'right_status'])

    labels_array = epochs.events[:, 2]
    return epochs, labels_array


def generate_epochs_visual_search(item_markers, item_markers_timestamps, event_markers, event_marker_timestamps,
                                  data_array,
                                  data_timestamps, data_channel_names,
                                  session_log, item_codes, tmin, tmax, event_ids, color_dict, title='',
                                  is_plotting=True):
    # interpolate nan's
    data_array = interpolate_array_nan(data_array)
    srate = 200
    eyetracking_with_event_marker_data = add_gaze_event_markers_to_data_array(item_markers,
                                                                              item_markers_timestamps,
                                                                              event_markers,
                                                                              event_marker_timestamps,
                                                                              data_array,
                                                                              data_timestamps, session_log,
                                                                              item_codes)

    info = mne.create_info(
        ['LSLTimestamp'] + data_channel_names + ['EventMarker'] + ["info1", "info2", "info3"],
        sfreq=srate,
        ch_types=['misc'] * (1 + len(data_channel_names)) + ['stim'] + [
            'misc'] * 3)  # with 3 additional info markers
    raw = mne.io.RawArray(eyetracking_with_event_marker_data, info)

    # pupil epochs
    epochs_pupil = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
                          tmax=tmax,
                          baseline=None,
                          preload=True,
                          verbose=False, picks=['left_pupil_size', 'right_pupil_size'])

    # Average epoch data
    if is_plotting:
        y_all = np.empty(shape=(0, epochs_pupil.get_data().shape[-1]))
        event_count_dict = {}
        for event_name, event_marker_id in event_ids.items():
            if event_name == 'Target' or event_name == 'Distractor':
                # if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
                y = epochs_pupil[event_name].get_data()
                # y = interpolate_epochs_nan(y)
                # y = np.mean(y, axis=1)  # average left and right
                y = y[:, 0, :]  # get the left eye data
                y_all = np.concatenate([y_all, y])
                event_count_dict[event_name] = len(y)
        y_all = scipy.stats.zscore(np.array(y_all), axis=1, ddof=0, nan_policy='propagate')

        previous_count = 0
        for event_name, event_count in event_count_dict.items():
            time_vector = np.linspace(tmin, tmax, y_all.shape[-1])
            y = np.array([mne.baseline.rescale(x, time_vector, (-0.1, 0.)) for x in
                          y_all[previous_count:(event_count + previous_count)]])
            previous_count = event_count
            y1 = np.mean(y, axis=0) + scipy.stats.sem(y, axis=0)  # this is the upper envelope
            y2 = np.mean(y, axis=0) - scipy.stats.sem(y, axis=0)  # this is the lower envelope
            plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                             interpolate=True,
                             alpha=0.5)
            plt.plot(time_vector, np.mean(y, axis=0), c=color_dict[event_name],
                     label='{0}, N={1}'.format(event_name, epochs_pupil[event_name].get_data().shape[0]))
        plt.xlabel('Time (sec)')
        plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
        plt.legend()
        plt.title(title)
        plt.show()

        # ERP Image
        for event_name, event_marker_id in event_ids.items():
            if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
                y = epochs_pupil[event_name].get_data()
                y = np.mean(y, axis=1)  # average left and right
                # y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')
                time_vector = np.linspace(tmin, tmax, y.shape[-1])
                plt.imshow(y)
                plt.xticks(np.arange(0, y.shape[1], y.shape[1] / 5),
                           ["{:6.2f}".format(x) for x in np.arange(tmin, tmax, (tmax - tmin) / 5)])
                plt.xlabel('Time (sec)')
                plt.ylabel('Trails')
                plt.legend()
                plt.title('{0}: {1}'.format(title, event_name))
                plt.show()

    epochs_gaze = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
                         tmax=tmax,
                         baseline=None,
                         preload=True,
                         verbose=False, picks=['raw_timestamp', 'status', 'gaze_forward_x', 'gaze_forward_y'])

    return epochs_pupil, epochs_gaze


def visualize_pupil_epochs(epochs, event_ids, tmin, tmax, color_dict, title, srate=200):
    for event_name, event_marker_id in event_ids.items():
        y = epochs[event_name].get_data()
        y = interpolate_epoch_zeros(y)  # remove nan
        y = interpolate_epochs_nan(y)  # remove nan
        assert np.sum(np.isnan(y)) == 0
        y = np.mean(y, axis=1)  # average left and right
        # y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

        y = np.mean(y, axis=0)
        y = y - y[int(abs(tmin) * srate)]  # baseline correct
        y1 = y + scipy.stats.sem(y, axis=0)  # this is the upper envelope
        y2 = y - scipy.stats.sem(y, axis=0)  # this is the lower envelope

        time_vector = np.linspace(tmin, tmax, y.shape[-1])
        plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                         interpolate=True,
                         alpha=0.5)
        plt.plot(time_vector, y, c=color_dict[event_name],
                 label='{0}, N={1}'.format(event_name, epochs[event_name].get_data().shape[0]))
    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
    plt.legend()
    plt.title(title)
    plt.show()


def generate_condition_sequence(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names,
                                session_log,
                                item_codes,
                                srate=200):  # use a fixed sampling rate for the sampling rate to match between recordings
    # interpolate nan's
    eyetracking_with_event_marker_data = add_eventMarkers_LSLTimestamp_to_data(event_markers,
                                                                               event_marker_timestamps,
                                                                               data_array,
                                                                               data_timestamps, session_log,
                                                                               item_codes)
    block_sequences = extract_block_data(eyetracking_with_event_marker_data, data_channel_names, 3, srate)
    return block_sequences


def flatten_list(l):
    return [item for sublist in l for item in sublist]
