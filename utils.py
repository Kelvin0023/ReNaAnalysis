import numpy as np
import scipy
from nslr_hmm import nslr_hmm
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


def add_event_markers_to_data_array(event_markers, event_ids, event_marker_timestamps, data_array, data_timestamps, session_log,
                                    item_codes):
    block_num = None
    assert event_markers.shape[0] == 4
    data_event_marker_array = np.zeros(shape=(4, data_array.shape[1]))

    for i in range(event_markers.shape[1]):
        event, info1, info2, info3 = event_markers[:, i]
        data_event_marker_index = (np.abs(data_timestamps - event_marker_timestamps[i])).argmin()

        if str(int(event)) in session_log.keys():
            print('Processing block with ID: {0}'.format(event))
            block_num = event
            data_event_marker_array[0][data_event_marker_index] = 4  # encodes start of a block
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

    return np.concatenate([data_array, data_event_marker_array], axis=0)


def generate_epochs(event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names, session_log,
                    item_codes, tmin, tmax, event_ids, color_dict, title=''):
    # interpolate nan's
    data_array_interpolated = interpolate_nan_array(data_array)
    # data_array_interpolated = data_array
    srate = len(data_timestamps) / (data_timestamps[-1] - data_timestamps[0])
    eyetracking_with_event_marker_data, event_ids = add_event_markers_to_data_array(event_markers,
                                                                                    event_marker_timestamps,
                                                                                    data_array_interpolated,
                                                                                    data_timestamps, session_log,
                                                                                    item_codes)

    info = mne.create_info(
        data_channel_names + ['EventMarker'] + ["CarouselDistance", "CarouselAngularSpeed", "CarouselIncidentAngle"],
        sfreq=srate,
        ch_types=['misc'] * len(data_channel_names) + ['stim'] + ['misc'] * 3)  # with 3 additional info markers
    raw = mne.io.RawArray(eyetracking_with_event_marker_data, info)

    # pupil epochs
    epochs = Epochs(raw, events=find_events(raw, stim_channel='EventMarker'), event_id=event_ids, tmin=tmin, tmax=tmax,
                    baseline=(0, 0),
                    preload=True,
                    verbose=False, picks=['L Pupil Diameter', 'R Pupil Diameter'])

    # Average epoch data
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

    # ERP Image
    for event_name, event_marker_id in event_ids.items():
        if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
            y = epochs[event_name].get_data()
            y = np.mean(y, axis=1)  # average left and right
            # y = scipy.stats.zscore(y, axis=1, ddof=0, nan_policy='propagate')
            time_vector = np.linspace(tmin, tmax, y.shape[-1])
            plt.imshow(y)
            plt.xticks(np.arange(0, y.shape[1], y.shape[1] / 5), ["{:6.2f}".format(x) for x in np.arange(tmin, tmax, (tmax-tmin)/5)])
            plt.xlabel('Time (sec)')
            plt.ylabel('Trails')
            plt.legend()
            plt.title('{0}: {1}'.format(title, event_name))
            plt.show()

    # gaze epochs
    COLORS = {
        nslr_hmm.FIXATION: 'blue',
        nslr_hmm.SACCADE: 'black',
        nslr_hmm.SMOOTH_PURSUIT: 'green',
        nslr_hmm.PSO: 'yellow',
    }

    NAMES = {
        nslr_hmm.FIXATION: 'Fixation',
        nslr_hmm.SACCADE: 'Saccade',
        nslr_hmm.SMOOTH_PURSUIT: 'Smooth pursuit',
        nslr_hmm.PSO: 'PSO',
    }
    epochs_gaze = Epochs(raw, events=find_events(raw), event_id=event_ids, tmin=0, tmax=tmax, baseline=(0, 0),
                    preload=True,
                    verbose=False, picks=['L Gaze Direction X', 'L Gaze Direction Y'])
    # iterate over the epochs
    for event_name, event_marker_id in event_ids.items():
        if event_name == 'Novelty' or event_name == 'Target' or event_name == 'Distractor':
            segment_class_array_dict = dict([(x, np.zeros(601)) for x in COLORS.keys()])
            gaze_epochs_data = epochs_gaze[event_name].get_data()
            # iterate over each epoch
            for eye in gaze_epochs_data:
                sample_class, segmentation, seg_class = nslr_hmm.classify_gaze(np.linspace(tmin, tmax, eye.shape[-1]), np.reshape(eye, newshape=[eye.shape[-1], -1]))
                # iterate over the classified segments
                for i, seg in enumerate(segmentation.segments):
                    cls = seg_class[i]
                    segment_class_array_dict[cls][np.array(seg.i)[0]:np.array(seg.i)[1]] += 1
            # normalize
            segment_array = np.array([x for x in segment_class_array_dict.values()])
            segment_array = (segment_array - segment_array.min())/(segment_array.max() - segment_array.min())
            for segment_class_ratio_data, segment_class_code in zip(segment_array, segment_class_array_dict.keys()):
                plt.plot(np.linspace(tmin, tmax, eye.shape[-1]), segment_class_ratio_data, label=NAMES[segment_class_code])
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.ylabel('Ratio of this gaze event occurs')
            plt.title('RSVP Gaze event ratio for participant 1 ' +event_name)
            plt.show()
    # epochs = Epochs(raw, events=find_events(raw), event_id=event_ids, tmin=0, tmax=tmax, baseline=(0, 0),
    #                 preload=True,
    #                 verbose=False,
    #                 picks=['L Gaze Direction X', 'L Gaze Direction Y', 'L Gaze Direction Z', 'R Gaze Direction X',
    #                        'R Gaze Direction Y', 'R Gaze Direction Z', "EventMarker", "CarouselDistance",
    #                        "CarouselAngularSpeed", "CarouselIncidentAngle"])
    #
    # y = epochs['Distractor'].get_data()
    # incidnet_angles = y[:, 9, 0]
    # sort_indices = np.argsort(incidnet_angles, axis=0)  # sort the ERP by incident angle
    # y = y[sort_indices]
    # y_gaze_x = np.mean(y[:, [0, 3], :], axis=1)  # take the gaze direction x and average
    # time_vector = np.linspace(tmin, tmax, y.shape[-1])
    # # [plt.plot(time_vector, x) for x in y_gaze_x]
    # plt.imshow(y_gaze_x)
    # plt.show()

    #
    # 1. grab the sequence for each block
    # 2. stitch the blocks together
    # 3. create design matrix based on the events

    plt.show()
    return epochs


def plot_epochs_visual_search(itemMarkers, itemMarkers_timestamps, event_markers, event_marker_timestamps, data_array, data_timestamps, data_channel_names,
                              session_log, item_codes, tmin, tmax, color_dict, title=''):
    # interpolate nan's
    data_array = interpolate_nan_array(data_array)
    srate = 200
    eyetracking_with_event_marker_data, event_ids = add_event_markers_to_data_array(event_markers,
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

