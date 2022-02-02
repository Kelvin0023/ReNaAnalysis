import numpy as np
import scipy
from scipy.interpolate import interp1d
import json

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import find_events, Epochs

from rena.utils.data_utils import RNStream


def interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
    interp = interp1d(indices[not_nan], x[not_nan])
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


def add_event_markers_to_data_array(event_markers, event_marker_timestamps, data_array, data_timestamps, session_log,
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
            print('    Item event {0} is novelty with info {1}'.format(event, str(data_event_marker_array[0:4,
                                                                                  data_event_marker_index])))

    return np.concatenate([np.expand_dims(data_timestamps, axis=0), data_array, data_event_marker_array], axis=0)
    # return np.concatenate(
    #     [np.expand_dims(data_timestamps, axis=0), data_array[1:, :], data_event_marker_array],
    #     axis=0)  # TODO because LSL cannot stream the raw nano second timestamp, we use the LSL timestamp instead. Convert the timestamps in second to nanosecond to comply with gaze detection code


def add_gaze_event_markers_to_data_array(item_markers, item_markers_timestamps, event_markers, event_marker_timestamps,
                                         data_array, data_timestamps,
                                         session_log, item_codes):
    block_num = None  # session log is keyed by the blocked_num
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

    data_block_starts_indices = np.argwhere(
        data_event_marker_array[0, :] == 4)  # start of a block is denoted by event marker 4
    data_block_ends_indices = np.argwhere(
        data_event_marker_array[0, :] == 5)  # end of a block is denoted by event marker 5

    # iterate through blocks
    for data_start_i, data_end_i in zip(data_block_starts_indices, data_block_ends_indices):
        # 1. find the event marker timestamps corresponding to the block start and end
        data_block_start_timestamp = data_timestamps[data_start_i]
        data_block_end_timestamp = data_timestamps[data_end_i]
        # 2. find the nearest timestamp of the block start and end in the item marker timestamps
        item_marker_block_start_index = np.argmin(np.abs(item_markers_timestamps - data_block_start_timestamp))
        item_marker_block_end_index = np.argmin(np.abs(item_markers_timestamps - data_block_end_timestamp))
        item_markers_of_block = item_markers[:, item_marker_block_start_index:item_marker_block_end_index]

        for i in range(30):
            this_item_marker = item_markers_of_block[i * 11: i * 11 + 1]
            # TODO finish this
        # 3. get the IsGazeRay intersected stream and their timestamps (item marker) keyed by the item count in block

        # 4. for each of the 30 items in the block, find where the IsGazeRay is true
        # 5. insert the gazed event marker in the data_event_marker_array at the data_timestamp nearest to the corresponding item_marker_timestamp

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
            print('    Item event {0} is novelty with info {1}'.format(event, str(data_event_marker_array[0:4,
                                                                                  data_event_marker_index])))

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


def generate_epochs(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names,
                    session_log,
                    item_codes, tmin, tmax, event_ids, color_dict, title='', is_plotting=True,
                    srate=200):  # use a fixed sampling rate for the sampling rate to match between recordings
    # interpolate nan's
    eyetracking_with_event_marker_data = add_event_markers_to_data_array(event_markers,
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

    epochs_gaze = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin,
                         tmax=tmax,
                         baseline=None,
                         preload=True,
                         verbose=False, picks=['raw_timestamp', 'status', 'gaze_forward_x', 'gaze_forward_y'])

    return epochs_pupil, epochs_gaze


def generate_epochs_visual_search(item_markers, item_markers_timestamps, event_markers, event_marker_timestamps,
                                  data_array,
                                  data_timestamps, data_channel_names,
                                  session_log, item_codes, tmin, tmax, event_ids, color_dict, title=''):
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
        data_channel_names + ['EventMarker'] + ["", "CarouselAngularSpeed", "CarouselIncidentAngle"],
        sfreq=srate,
        ch_types=['misc'] * len(data_channel_names) + ['stim'] + ['misc'] * 3)  # with 3 additional info markers
    raw = mne.io.RawArray(eyetracking_with_event_marker_data, info)

    # pupil epochs
    epochs = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin, tmax=tmax,
                    baseline=(0, 0),
                    preload=True,
                    verbose=False, picks=['L Pupil Diameter', 'R Pupil Diameter'])

    for event_name, event_marker_id in event_ids.items():
        if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
            y = epochs[event_name].get_data()
            y = np.mean(y, axis=1)  # average left and right
            y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

            y1 = np.mean(y, axis=0) + scipy.stats.sem(y, axis=0)  # this is the upper envelope
            y2 = np.mean(y, axis=0) - scipy.stats.sem(y, axis=0)  # this is the lower envelope
            time_vector = np.linspace(tmin, tmax, y.shape[-1])
            plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                             interpolate=True,
                             alpha=0.5)
            plt.plot(time_vector, np.mean(y, axis=0), c=color_dict[event_name],
                     label='{0}, N={1}'.format(event_name, epochs[event_name].get_data().shape[0]))
    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right z-score), shades are SEM')
    plt.legend()
    plt.title(title)
    plt.show()

    # gaze epochs
    # epochs = Epochs(raw, events=find_events(raw), event_id=event_ids, tmin=0, tmax=tmax, baseline=(0, 0),
    #                 preload=True,
    #                 verbose=False, picks=['L Gaze Direction X', 'L Gaze Direction Y', 'L Gaze Direction Z', 'R Gaze Direction X', 'R Gaze Direction Y', 'R Gaze Direction Z', "EventMarker", "CarouselDistance", "CarouselAngularSpeed", "CarouselIncidentAngle"])
    # y = epochs['Distractor'].get_data()
    # incidnet_angles = y[:, 9, 0]
    # sort_indices = np.argsort(incidnet_angles, axis=0)  # sort the ERP by incident angle
    # y = y[sort_indices]
    # y_gaze_x = np.mean(y[:, [0, 3], :], axis=1)  # take the gaze direction x and average
    # time_vector = np.linspace(tmin, tmax, y.shape[-1])
    # # [plt.plot(time_vector, x) for x in y_gaze_x]
    # plt.imshow(y_gaze_x)
    # plt.show()
    return epochs


def visualize_epochs(epochs, event_ids, tmin, tmax, color_dict, title):
    for event_name, event_marker_id in event_ids.items():
        if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
            y = epochs[event_name].get_data()
            y = np.mean(y, axis=1)  # average left and right
            y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')

            y1 = np.mean(y, axis=0) + scipy.stats.sem(y, axis=0)  # this is the upper envelope
            y2 = np.mean(y, axis=0) - scipy.stats.sem(y, axis=0)  # this is the lower envelope
            time_vector = np.linspace(tmin, tmax, y.shape[-1])
            plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                             interpolate=True,
                             alpha=0.5)
            plt.plot(time_vector, np.mean(y, axis=0), c=color_dict[event_name],
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
    eyetracking_with_event_marker_data = add_event_markers_to_data_array(event_markers,
                                                                         event_marker_timestamps,
                                                                         data_array,
                                                                         data_timestamps, session_log,
                                                                         item_codes)
    block_sequences = extract_block_data(eyetracking_with_event_marker_data, data_channel_names, 3, srate)
    return block_sequences
