import math

import numpy as np
import scipy
from matplotlib import pyplot as plt

from eye.EyeUtils import temporal_filter_fixation
from params import *
from utils.Event import Event, add_event_meta_info, get_events_between, is_event_in_block, copy_item_info, \
    get_overlapping_events
from copy import copy


class Saccade(Event):
    def __init__(self, amplitude, duration, peak_velocity, average_velocity, onset, offset, onset_time, offset_time,
                 peak, detection_alg, *args, **kwargs):
        super().__init__(onset_time, *args, **kwargs)
        self.amplitude = amplitude
        self.duration = duration
        self.peak_velocity = peak_velocity
        self.average_velocity = average_velocity
        self.from_stim = None  # to what stimulus is the saccade directed
        self.to_stim = None  # to what stimulus is the saccade directed
        self.onset = onset
        self.offset = offset
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.peak = peak
        self.detection_alg = detection_alg
        self.epoched = False


class Fixation(Event):
    def __init__(self, duration, dispersion, preceding_saccade, following_saccade, onset, offset, onset_time,
                 offset_time, detection_alg, *args, **kwargs):
        super().__init__(onset_time, *args, **kwargs)
        self.duration = duration
        self.dispersion = dispersion
        self.stim = None  # at what stimulus is the participant fixated on
        self.preceding_saccade = preceding_saccade
        self.following_saccade = following_saccade
        self.onset = onset
        self.offset = offset
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.detection_alg = detection_alg
        self.epoched = False


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def gaze_event_detection_I_VT(eyetracking_data_timestamps, events, headtracking_data_timestamps=None):
    eyetracking_data, eyetracking_timestamps = eyetracking_data_timestamps
    assert eyetracking_data.shape[0] == len(varjoEyetracking_preset['ChannelNames'])
    varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']
    gaze_xy = eyetracking_data[[varjoEyetracking_channelNames.index('gaze_forward_{0}'.format(x)) for x in ['x', 'y']]]
    gaze_status = eyetracking_data[varjoEyetracking_channelNames.index('status')]

    head_rotation_xy_eyesampled = None
    if headtracking_data_timestamps is not None:
        print("Processing headtracking data")
        headtracking_data, head_tracker_timestamps = headtracking_data_timestamps
        yaw_pitch_indices = headtracker_preset['ChannelNames'].index("Head Yaw"), headtracker_preset['ChannelNames'].index("Head Pitch")
        head_rotation_xy = headtracking_data[yaw_pitch_indices, :]
        # find the closest head tracking data point for each eyetracking point
        head_rotation_xy_eyesampled = scipy.signal.resample(head_rotation_xy, len(eyetracking_timestamps),axis=1)

    gaze_behavior_events, fixations, saccades, velocity = gaze_event_detection(gaze_xy,
                                                                               gaze_timestamps=eyetracking_timestamps,
                                                                               gaze_status=gaze_status,
                                                                               head_rotation=head_rotation_xy_eyesampled)
    fixations, saccades = add_event_info_to_gaze(fixations, events)
    return fixations + saccades


def add_event_info_to_gaze(fixations, events):
    """
    process a dataset that has gaze behavior marker and gaze marker (gaze ray intersect with object), use the gaze marker to find
    first identify the gaze marker inside a fixation,
    if there is an gaze marker, find what stimulus type is that gaze marker
        then we set the preceding saccade's destination to the stimulus type
        then we set the following saccade's origin to the stimulus type
    if there is no gaze marker, then this fixation is not on an object

    note the function can run on either exg or eyetracking, we use exg here as it has higher sampling rate and gives
    presumably better synchronization
    @rtype: new lists of fixations and saccades
    """
    fix_in_block = []
    sac_in_block = []
    for f in fixations:
        if is_event_in_block(f, events):
            f = add_event_meta_info(f, events)
            f.preceding_saccade = add_event_meta_info(f, events)
            # gaze_intersect_events = get_events_between(f.onset_time, f.offset_time, events, lambda x: x.gaze_intersect is not None )
            overlapping_gaze_intersects = get_overlapping_events(f.onset_time, f.offset_time, events, lambda x: type(x) == GazeRayIntersect)

            if len(overlapping_gaze_intersects) > 0:
                e = overlapping_gaze_intersects[0]  # IMPORTANT pick the first gaze event
                f = copy_item_info(f, e)
                f.preceding_saccade = copy_item_info(f.preceding_saccade, e)
            else:
                f.preceding_saccade.dtn = dtnn_types['Null']

            fix_in_block.append(f)
            sac_in_block.append(f.preceding_saccade)
    return fix_in_block, sac_in_block


def gaze_event_detection(gaze_xy, gaze_timestamps, gaze_xy_format="ratio", gaze_status=None,
                         saccade_min_peak=6, saccade_min_amplitude=2, saccade_spacing=20e-3, saccade_min_sample=2,
                         fixation_min_sample=2, glitch_threshold=1000, head_rotation=None):
    """
    gaze event detection based on Velocity Threshold Identification (I-VT)
    inspirations:
    https://journals.physiology.org/doi/abs/10.1152/jn.00237.2011
    https://dl.acm.org/doi/pdf/10.1145/1743666.1743682 for a comparison between different gaze event classification methods
    https://ieeexplore.ieee.org/abstract/document/8987791/ for some implementation of I-VT method
    currently no criterion is applied to the fixations

    @param gaze_status: if the gaze is valid
    @param saccade_min_peak: float: the peak velocity of a potential saccade period has to be greater than this minimal
    to be considered a saccade, unit is in deg/s
    @param saccade_min_amplitude: float: the minimal amplitude of for a saccade, unit is deg
    @param saccade_spacing: float: the minimal amplitude of for a saccade, unit is second, default to 20ms according to https://iovs.arvojournals.org/article.aspx?articleid=2193271
    @param glitch_threshold: float: unit in deg/s, time points with velocity exceeding this threshold will be considered
    a glitch in the recording
    @return
    event types: -1: noise or glitch; 1: saccade; 2: fixation
    """
    events = np.zeros(gaze_timestamps.shape)
    if gaze_status is not None: events[
        gaze_status != 2] = -1  # remove points where the status is invalid from the eyetracker
    saccades = []  # items are tuple of three: saccade onset index, saccade peak velocity index, saccade offset index, Saccade object,
    fixations = []  # items are tuple of three: saccade onset index, saccade peak velocity index, saccade offset index, Saccade object,

    gaze_xy_deg = (180 / math.pi) * np.arcsin(gaze_xy) if gaze_xy_format == 'ratio' else gaze_xy

    detection_alg = 'I-DT' if head_rotation is None else 'I-DT-Head'
    if head_rotation is not None:
        gaze_xy_deg = gaze_xy_deg + head_rotation


    # calculate eye velocity in degrees
    dxy = np.diff(gaze_xy_deg, axis=1, prepend=gaze_xy_deg[:, :1])
    dtheta = np.linalg.norm(dxy, axis=0)

    try:
        assert not np.any(np.diff(gaze_timestamps, prepend=1) == 0)
    except AssertionError:
        raise ValueError('Invalid gaze timestamps, time delta is zero at {0}'.format(np.argwhere(np.diff(gaze_timestamps, prepend=1) == 0)))
    velocities = dtheta / np.diff(gaze_timestamps, prepend=1)
    velocities[0] = 0.  # assume the first velocity is 0

    events[velocities > glitch_threshold] = -1

    acceleration = np.diff(velocities, prepend=velocities[0])
    acceleration_zero_crossings = np.where(np.diff(np.sign(acceleration)))[0]
    for crossing_i in range(0, len(acceleration_zero_crossings) - 2):
        onset = acceleration_zero_crossings[crossing_i]
        peak = acceleration_zero_crossings[crossing_i + 1]
        offset = acceleration_zero_crossings[crossing_i + 2]

        if not acceleration[peak] > 0 and acceleration[peak + 1] < 0:
            # check the peak is cross from positive to negative
            continue
        if len(saccades) > 0 and gaze_timestamps[onset] - gaze_timestamps[
            saccades[-1].offset] < saccade_spacing:  # check temporal spacing condition for saccade
            continue
        if np.any(events[onset:offset] == -1):  # check if gaze status is invalid during the potential saccade
            continue
        if offset - onset < saccade_min_sample:
            continue

        amplitude = np.linalg.norm(gaze_xy_deg[:, offset] - gaze_xy_deg[:, onset], axis=0)
        peak_velocity = velocities[peak]
        average_velocity = np.mean(velocities[onset:offset])
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]
        if velocities[peak] > saccade_min_peak and amplitude > saccade_min_amplitude:
            saccades.append(
                Saccade(amplitude, duration, peak_velocity, average_velocity, onset, offset, gaze_timestamps[onset],
                        gaze_timestamps[offset], peak, detection_alg=detection_alg))

    # identify the fixations for all the intervals between saccades
    fixation_inteval_indices = [(saccades[i - 1].offset, saccades[i].onset) for i in
                                range(1, len(saccades))]  # IGNORE the interval before the first saccade
    for i, (onset, offset) in enumerate(fixation_inteval_indices):
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]
        _xy_deg = gaze_xy_deg[:, onset:offset][:,
                  events[onset:offset] != -1]  # check the dispersion excluding the invalid points
        if _xy_deg.shape[
            1] != 0 and offset - onset > fixation_min_sample:  # if the entire interval is invalid, then we do NOT add it to fixation
            dispersion = np.max(_xy_deg, axis=1) - np.min(_xy_deg, axis=1)
            fixations.append(
                Fixation(duration, dispersion, saccades[i], saccades[i + 1], onset, offset, gaze_timestamps[onset],
                         gaze_timestamps[offset], detection_alg=detection_alg))

    glitch_precentage = np.sum(events == -1) / len(events)

    for s in saccades:
        events[s.onset:s.offset] = SACCADE_CODE
    for f in fixations:
        events[f.onset:f.offset] = FIXATION_CODE
    print("Detected {} fixation from I-DT with {}% glitch percentage {}".format(len(fixations), glitch_precentage * 100, "" if head_rotation is None else "with Head rotation"))
    return events, fixations, saccades, velocities


def plot_gaze_events_overlay(start_time, end_time, gaze_timestamps, saccades, fixations, velocities):
    """

    @param start_time: time in seconds
    @param end_time: time in seconds
    """
    try:
        assert start_time < np.max(gaze_timestamps) and start_time > np.min(gaze_timestamps)
        assert end_time < np.max(gaze_timestamps) and end_time > np.min(gaze_timestamps)
        assert end_time > start_time
    except AssertionError:
        raise AttributeError("Invalid start and end")
    plt.rcParams["figure.figsize"] = (20, 10)
    saccades_in_bounds = [s for s in saccades if s.onset_time > start_time and s.offset_time < end_time]
    fixations_in_bounds = [f for f in fixations if f.onset_time > start_time and f.offset_time < end_time]

    start_index = np.argmin(np.abs(gaze_timestamps - start_time))
    end_index = np.argmin(np.abs(gaze_timestamps - end_time))
    plt.plot(gaze_timestamps[start_index:end_index], velocities[start_index:end_index], label='velocity')

    plt.axvspan(gaze_timestamps[saccades_in_bounds[0].onset], gaze_timestamps[saccades_in_bounds[0].offset], alpha=0.5, color='r', label='saccade')
    plt.axvspan(gaze_timestamps[fixations_in_bounds[0].onset], gaze_timestamps[fixations_in_bounds[0].offset], alpha=0.5, color='g', label='fixation')
    for s in saccades_in_bounds[1:]:
        plt.axvspan(gaze_timestamps[s.onset], gaze_timestamps[s.offset], alpha=0.5, color='r')
    for f in fixations_in_bounds[1:]:
        plt.axvspan(gaze_timestamps[f.onset], gaze_timestamps[f.offset], alpha=0.5, color='g')
    plt.ylim(0, 1000)
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (deg/sec)')
    plt.legend()
    plt.show()


def gaze_event_detection_PatchSim(ps_fixation_detection_data, ps_fixation_detection_timestamps, events):
    fix_list_filtered = temporal_filter_fixation(ps_fixation_detection_data[1], marker_mode='marker', verbose=0)

    onset_timestamps = ps_fixation_detection_timestamps[fix_list_filtered == 1]
    offet_timestamps = ps_fixation_detection_timestamps[fix_list_filtered == 2]

    fixation_events = []
    for onset_ts, offset_ts in zip(onset_timestamps, offet_timestamps):
        f = Fixation(offset_ts - onset_ts, None, None, None, None, None, onset_ts, offset_ts, "Patch-Sim")
        overlapping_gaze_intersects = get_overlapping_events(f.onset_time, f.offset_time, events, lambda x: type(x) == GazeRayIntersect)
        if len(overlapping_gaze_intersects) > 0:
            e = overlapping_gaze_intersects[0]  # IMPORTANT pick the first gaze event
            f = copy_item_info(f, e)
        else:
            f.dtn = dtnn_types['Null']
        fixation_events.append(f)
    print('Detected {} fixations from Patch similarity based fixation detection'.format(
        int(np.sum(fix_list_filtered[fix_list_filtered == 1]))))
    return fixation_events


class GazeRayIntersect(Event):
    def __init__(self, timestamp, onset_time, offset_time, *args, **kwargs):
        super().__init__(timestamp, *args, **kwargs)
        self.onset_time = onset_time
        self.offset_time = offset_time
