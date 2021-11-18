import numpy as np
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


def interpolate_nan_array(data_array):
    """
    :param data_array: channel first, time last
    """
    return np.array([interpolate_nan(x) for x in data_array])

def add_event_markers_to_data_array(event_markers, event_marker_timestamps, data_array, data_timestamps, session_log, item_codes):
    block_num = None
    data_event_marker_array = np.zeros(shape=(1, data_array.shape[1]))
    event_ids = {'Novelty': 3, 'Target': 2, 'Distractor': 1}

    for i, event in enumerate(event_markers):
        if str(int(event)) in session_log.keys():
            print('Processing block with ID: {0}'.format(event))
            block_num = event
            continue
        if event in item_codes:
            targets = session_log[str(int(block_num))]['targets']
            distractors = session_log[str(int(block_num))]['distractors']
            novelties = session_log[str(int(block_num))]['novelties']

            data_event_marker_index = (np.abs(data_timestamps - event_marker_timestamps[i])).argmin()
            if event in distractors:
                data_event_marker_array[0][data_event_marker_index] = 1
                print('    Item event {0} is distractor'.format(event))
            elif event in targets:
                data_event_marker_array[0][data_event_marker_index] = 2
                print('    Item event {0} is target'.format(event))
            elif event in novelties:
                data_event_marker_array[0][data_event_marker_index] = 3
                print('    Item event {0} is novelty'.format(event))
    return np.concatenate([data_array, data_event_marker_array], axis=0), event_ids


def plot_epochs(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names, session_log, item_codes, tmin, tmax, color_dict, title=''):
    # interpolate nan's
    data_array = interpolate_nan_array(data_array)

    srate = len(data_timestamps) / (data_timestamps[-1] - data_timestamps[0])
    eyetracking_event_marker_data, event_ids = add_event_markers_to_data_array(event_markers, event_marker_timestamps,
                                                                               data_array,
                                                                               data_timestamps, session_log,
                                                                               item_codes)

    info = mne.create_info(data_channel_names + ['EventMarker'], sfreq=srate,
                           ch_types=['misc'] * len(data_channel_names) + ['stim'])
    raw = mne.io.RawArray(eyetracking_event_marker_data, info)
    epochs = Epochs(raw, events=find_events(raw), event_id=event_ids, tmin=tmin, tmax=tmax, baseline=(None, 0),
                    preload=True,
                    verbose=False, picks=['L Pupil Diameter', 'R Pupil Diameter'])

    for event_name, event_marker_id in event_ids.items():
        y = epochs[event_name].get_data()
        y = np.mean(y, axis=1)
        y1 = np.mean(y, axis=0) + 0.075 * np.std(y, axis=0)  # this is the upper envelope
        y2 = np.mean(y, axis=0) - 0.075 * np.std(y, axis=0)  # this is the lower envelope
        time_vector = np.linspace(tmin, tmax, y.shape[-1])
        plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=color_dict[event_name],
                         interpolate=True,
                         alpha=0.5)
        plt.plot(time_vector, np.mean(y, axis=0), c=color_dict[event_name], label=event_name)

    plt.xlabel('Time (sec)')
    plt.ylabel('Pupil Diameter (averaged left and right in m)')
    plt.legend()
    plt.title(title)
    plt.show()
    return epochs