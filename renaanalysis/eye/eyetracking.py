import math

import numpy as np
import scipy
from matplotlib import pyplot as plt

from renaanalysis.eye.EyeUtils import temporal_filter_fixation
from renaanalysis.params.params import dtnn_types, SACCADE_CODE, \
    FIXATION_CODE, varjoEyetracking_chs, headtracker_chs
from renaanalysis.utils.Event import Event, add_event_meta_info, get_events_between, is_event_in_block, copy_item_info, get_overlapping_events
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
        self.is_first_long_gaze = False


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps=None):
    eyetracking_data, eyetracking_timestamps = eyetracking_data_timestamps
    assert eyetracking_data.shape[0] == len(varjoEyetracking_chs)
    gaze_xyz = eyetracking_data[[varjoEyetracking_chs.index('gaze_forward_{0}'.format(x)) for x in ['x', 'y', 'z']]]
    gaze_status = eyetracking_data[varjoEyetracking_chs.index('status')]

    head_rotation_xy_degree_eyesampled = None
    if headtracking_data_timestamps is not None:
        print("Processing headtracking data")
        headtracking_data, head_tracker_timestamps = headtracking_data_timestamps
        yaw_pitch_indices = headtracker_chs.index("Head Yaw"), headtracker_chs.index("Head Pitch")
        head_rotation_xy = headtracking_data[yaw_pitch_indices, :]
        # find the closest head tracking data point for each eyetracking point
        head_rotation_xy_degree_eyesampled = scipy.signal.resample(head_rotation_xy, len(eyetracking_timestamps),axis=1)

    return eyetracking_timestamps, gaze_xyz, gaze_status, head_rotation_xy_degree_eyesampled
def gaze_event_detection_I_DT(eyetracking_data_timestamps, events, headtracking_data_timestamps=None):
    eyetracking_timestamps, gaze_xyz, gaze_status, head_rotation_xy_eyesampled = _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps)
    gaze_behavior_events, fixations, saccades, velocity = fixation_detection_i_dt(gaze_xyz,
                                                                                  gaze_timestamps=eyetracking_timestamps,
                                                                                  gaze_status=gaze_status,
                                                                                  head_rotation_xy_degree=head_rotation_xy_eyesampled)
    fixations, saccades = add_event_info_to_gaze(fixations, events)
    print(', {} fixations are first long on objects. Targets: {}, Distractors {}'.format(len([f for f in fixations if f.is_first_long_gaze]), len([f for f in fixations if f.dtn==1]), len([f for f in fixations if f.dtn==2])))
    return fixations + saccades

def gaze_event_detection_I_VT(eyetracking_data_timestamps, events, headtracking_data_timestamps=None):
    eyetracking_timestamps, gaze_xyz, gaze_status, head_rotation_xy_eyesampled = _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps)

    gaze_behavior_events, fixations, saccades, velocity = fixation_detection_i_vt(gaze_xyz,
                                                                                  gaze_timestamps=eyetracking_timestamps,
                                                                                  gaze_status=gaze_status,
                                                                                  head_rotation_xy_degree=head_rotation_xy_eyesampled)
    fixations, saccades = add_event_info_to_gaze(fixations, events)
    print(', {} fixations are first long on objects. Targets: {}, Distractors {}'.format(len([f for f in fixations if f.is_first_long_gaze]), len([f for f in fixations if f.dtn==1]), len([f for f in fixations if f.dtn==2])))
    return fixations + saccades


def add_event_info_to_gaze(fixations, events, long_gaze_threshold=0.15):
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

    @TODO this is a analysis bottleneck function make this multi-process
    """
    fix_in_block = []
    sac_in_block = []
    gaze_intersects = [x for x in events if type(x) == GazeRayIntersect]
    fixated = []
    for i, f in enumerate(fixations):
        if is_event_in_block(f, events):
            f = add_event_meta_info(f, events)
            try:
                if len(fix_in_block) != 0:  # ignore the first saccade in block
                    f.preceding_saccade = add_event_meta_info(f.preceding_saccade, events)
            except ValueError as e:
                if i == 0:  # the saccade of the first fixation maybe outside of the block
                    f.preceding_saccade.block_condition = f.block_condition
                    f.preceding_saccade.is_practice = f.is_practice
                    f.preceding_saccade.block_id = f.block_id
                else: raise e
                    # gaze_intersect_events = get_events_between(f.onset_time, f.offset_time, events, lambda x: x.gaze_intersect is not None )
            overlapping_gaze_intersects = get_overlapping_events(f.onset_time, f.offset_time, gaze_intersects)

            if len(overlapping_gaze_intersects) > 0:
                e = overlapping_gaze_intersects[0]  # IMPORTANT pick the first gaze event
                f = copy_item_info(f, e)
                if (f.item_index, f.block_id) not in fixated and f.duration > long_gaze_threshold:
                    f.is_first_long_gaze = True
                    fixated.append((f.item_index, f.block_id))
                f.preceding_saccade = copy_item_info(f.preceding_saccade, e)
            else:
                f.preceding_saccade.dtn = dtnn_types['Null']

            fix_in_block.append(f)
            sac_in_block.append(f.preceding_saccade)
    return fix_in_block, sac_in_block


def _calculate_gaze_angles(gaze_vector, head_rotation_xy_degree):
    gaze_yaw_degree = np.rad2deg(np.arctan2(gaze_vector[0, :], gaze_vector[2, :]))
    gaze_pitch_degree = np.rad2deg(np.arctan2(gaze_vector[1, :], gaze_vector[2, :]))

    # Calculate the angle of the 2D vector
    gaze_angles_degree = np.rad2deg(np.arctan2(gaze_vector[1, :], gaze_vector[0, :]))

    if head_rotation_xy_degree is not None:
        head_rotation_xy_degree_diff = np.diff(head_rotation_xy_degree, prepend=head_rotation_xy_degree[:, 0][..., np.newaxis])
        gaze_yaw_degree = gaze_yaw_degree + head_rotation_xy_degree_diff[0, :]
        gaze_pitch_degree = gaze_pitch_degree + head_rotation_xy_degree_diff[1, :]

        head_yaw_rad = np.deg2rad(head_rotation_xy_degree[0, :])
        head_pitch_rad = np.deg2rad(head_rotation_xy_degree[1, :])
        x = np.sin(head_yaw_rad) * np.cos(head_pitch_rad)
        y = np.sin(head_pitch_rad)
        z = np.cos(head_yaw_rad) * np.cos(head_pitch_rad)
        vector = np.array([x, y, z])

        gaze_angles_degree = gaze_angles_degree + np.rad2deg(np.arctan2(vector[1, :], vector[0, :]))

    # delta_gaze_angle_degree = np.diff(gaze_angles_degree, prepend=gaze_angles_degree[0])
    return gaze_yaw_degree, gaze_pitch_degree, gaze_angles_degree


def i_dt_is_fixation(gaze_angles_degree_glitched_nan, gaze_point_window_start_index, gaze_point_window_end_index, dispersion_threshold_degree):
    gaze_angle_window_diff = np.diff(gaze_angles_degree_glitched_nan[gaze_point_window_start_index:gaze_point_window_end_index])
    return not np.any(gaze_angle_window_diff[np.logical_not(np.isnan(gaze_angle_window_diff))] > dispersion_threshold_degree)

def fixation_detection_i_dt(gaze_xyz, gaze_timestamps, gaze_xy_format="ratio", gaze_status=None,
                            fixation_min_duraiton_second=0.1, fixation_velocity_threshold=30, dispersion_threshold_degree=0.5,
                            glitch_threshold=1000, saccade_min_sample=2, head_rotation_xy_degree=None):
    '''
    gaze event detection based on Displacement Threshold Identification (I-DT)
    @param gaze_xy:
    @param gaze_timestamps:
    @param gaze_xy_format:
    @param gaze_status:
    @param saccade_min_peak:
    @param saccade_min_amplitude:
    @param saccade_spacing:
    @param saccade_min_sample:
    @param fixation_min_sample:
    @param glitch_threshold:
    @param head_rotation:
    @return:
    '''
    events = np.zeros(gaze_timestamps.shape)

    gaze_yaw_degree, gaze_pitch_degree, gaze_angles_degree = _calculate_gaze_angles(gaze_xyz, head_rotation_xy_degree)

    velocities = np.gradient(gaze_angles_degree) / np.diff(gaze_timestamps, prepend=1)
    velocities[0] = 0.  # assume the first velocity is 0

    events[velocities > glitch_threshold] = -1

    fixations = []
    saccades = []
    detection_alg = 'I-DT' if head_rotation_xy_degree is None else 'I-DT-Head'

    gaze_point_window_start_index = 0
    gaze_point_window_end_index = 1

    gaze_angles_degree_glitched_nan = np.copy(gaze_angles_degree)
    gaze_angles_degree_glitched_nan[velocities > glitch_threshold] = np.nan
    while (gaze_point_window_end_index < len(gaze_timestamps)):
        print(f"Running i-dt fixation detection, window start is at {gaze_point_window_start_index}, total length is {len(gaze_timestamps)}", end='\r')
        # for timestamp, yaw, pitch, gaze_angle, velocity in zip(gaze_timestamps, gaze_yaw_degree, gaze_pitch_degree, gaze_angles_degree, velocities):
        # Check if this is the first gaze point in the sequence
        # if last_gaze_point is None:
        #     last_gaze_point = (timestamp, yaw, pitch)
        #     continue
        # yaw_diff = yaw - last_gaze_point[1]
        # pitch_diff = pitch - last_gaze_point[2]
        while gaze_timestamps[gaze_point_window_end_index] - gaze_timestamps[gaze_point_window_start_index] < fixation_min_duraiton_second:
            gaze_point_window_end_index += 1
            if gaze_point_window_end_index >= len(gaze_timestamps):
                break

        if not i_dt_is_fixation(gaze_angles_degree_glitched_nan, gaze_point_window_start_index, gaze_point_window_end_index, dispersion_threshold_degree):
            gaze_point_window_start_index += 1
        else:
            gaze_point_window_end_index += 1
            while i_dt_is_fixation(gaze_angles_degree_glitched_nan, gaze_point_window_start_index, gaze_point_window_end_index, dispersion_threshold_degree):
                gaze_point_window_end_index += 1
                if gaze_point_window_end_index >= len(gaze_timestamps):
                    break
            gaze_point_window_end_index -= 1

            if len(fixations) > 0:
                last_fixation = fixations[-1]
                amplitude = gaze_angles_degree[gaze_point_window_start_index] - gaze_angles_degree[last_fixation.offset]
                duration = gaze_timestamps[gaze_point_window_start_index] - gaze_timestamps[last_fixation.offset]
                saccade_velocities = velocities[last_fixation.offset:gaze_point_window_start_index][events[last_fixation.offset:gaze_point_window_start_index] != -1]
                if len(saccade_velocities) > 0:
                    peak = np.argmax(saccade_velocities)
                    peak_velocity = np.max(saccade_velocities)
                    average_velocity = np.mean(saccade_velocities)
                else:
                    peak = np.nan
                    peak_velocity = np.nan
                    average_velocity = np.nan
                this_saccade = Saccade(amplitude, duration, peak_velocity, average_velocity, last_fixation.offset, gaze_point_window_start_index,
                            gaze_timestamps[last_fixation.offset],
                            gaze_timestamps[gaze_point_window_start_index], peak, detection_alg=detection_alg)
                saccades.append(this_saccade)
                fixations[-1].following_saccade = this_saccade
            else:
                this_saccade = None
            if gaze_point_window_end_index >= len(gaze_timestamps):
                break

            fixation_duration = gaze_timestamps[gaze_point_window_end_index] - gaze_timestamps[gaze_point_window_start_index]
            _yaw_deg = gaze_yaw_degree[gaze_point_window_start_index:gaze_point_window_end_index][events[gaze_point_window_start_index:gaze_point_window_end_index] != -1]   # check the dispersion excluding the invalid points
            _pitch_deg = gaze_pitch_degree[gaze_point_window_start_index:gaze_point_window_end_index][events[gaze_point_window_start_index:gaze_point_window_end_index] != -1]   # check the dispersion excluding the invalid points
            _xy_deg = np.stack((_yaw_deg, _pitch_deg), axis=1)
            dispersion = np.max(_xy_deg, axis=0) - np.min(_xy_deg, axis=0)

            fixations.append(Fixation(fixation_duration, dispersion, this_saccade, None,
                                      gaze_point_window_start_index, gaze_point_window_end_index,
                                      gaze_timestamps[gaze_point_window_start_index],gaze_timestamps[gaze_point_window_end_index], detection_alg=detection_alg))
            gaze_point_window_end_index += saccade_min_sample  # add the min saccade samples
            gaze_point_window_start_index = gaze_point_window_end_index

    glitch_precentage = np.sum(events == -1) / len(events)
    for s in saccades:
        events[s.onset:s.offset] = SACCADE_CODE
    for f in fixations:
        events[f.onset:f.offset] = FIXATION_CODE
    print("Detected {} fixation from I-DT with {}% glitch percentage {}".format(len(fixations), glitch_precentage * 100, "" if head_rotation_xy_degree is None else "with head rotation"))
    return events, fixations, saccades, velocities

def fixation_detection_i_vt(gaze_xyz, gaze_timestamps, gaze_xy_format="ratio", gaze_status=None,
                            saccade_min_peak=6, saccade_min_amplitude=2, saccade_spacing=20e-3, saccade_min_sample=2,
                            fixation_min_sample=2, glitch_threshold=1000, head_rotation_xy_degree=None):
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

    gaze_yaw_degree, gaze_pitch_degree, gaze_angles_degree = _calculate_gaze_angles(gaze_xyz, head_rotation_xy_degree)
    detection_alg = 'I-VT' if head_rotation_xy_degree is None else 'I-VT-Head'

    try:
        assert not np.any(np.diff(gaze_timestamps, prepend=1) == 0)
    except AssertionError:
        raise ValueError('Invalid gaze timestamps, time delta is zero at {0}'.format(np.argwhere(np.diff(gaze_timestamps, prepend=1) == 0)))
    velocities = np.gradient(gaze_angles_degree) / np.diff(gaze_timestamps, prepend=1)
    velocities[0] = 0.  # assume the first velocity is 0

    events[velocities > glitch_threshold] = -1

    acceleration = np.gradient(velocities) / np.diff(gaze_timestamps, prepend=1)
    acceleration_zero_crossings = np.where(np.diff(np.sign(acceleration)))[0]
    for crossing_i in range(0, len(acceleration_zero_crossings) - 2):
        onset = acceleration_zero_crossings[crossing_i]
        peak = acceleration_zero_crossings[crossing_i + 1]
        offset = acceleration_zero_crossings[crossing_i + 2]

        # if not acceleration[peak] > 0 and acceleration[peak + 1] < 0:
        #     # check the peak is cross from positive to negative
        #     continue
        if len(saccades) > 0 and gaze_timestamps[onset] - gaze_timestamps[saccades[-1].offset] < saccade_spacing:  # check temporal spacing condition for saccade
            continue
        if np.any(events[onset:offset] == -1):  # check if gaze status is invalid during the potential saccade
            continue
        if offset - onset < saccade_min_sample:
            continue

        amplitude = gaze_angles_degree[offset] - gaze_angles_degree[onset]

        peak_velocity = velocities[peak]
        average_velocity = np.mean(velocities[onset:offset])
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]
        if velocities[peak] > saccade_min_peak and amplitude > saccade_min_amplitude:
            saccades.append(
                Saccade(amplitude, duration, peak_velocity, average_velocity, onset, offset, gaze_timestamps[onset], gaze_timestamps[offset], peak, detection_alg=detection_alg))

    # identify the fixations for all the intervals between saccades
    fixation_interval_indices = [(saccades[i - 1].offset, saccades[i].onset) for i in range(1, len(saccades))]  # IGNORE the interval before the first saccade
    for i, (onset, offset) in enumerate(fixation_interval_indices):
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]

        # _xy_deg = dtheta_degree[:, onset:offset][:, events[onset:offset] != -1] -    # check the dispersion excluding the invalid points
        _yaw_deg = gaze_yaw_degree[onset:offset][events[onset:offset] != -1]   # check the dispersion excluding the invalid points
        _pitch_deg = gaze_pitch_degree[onset:offset][events[onset:offset] != -1]   # check the dispersion excluding the invalid points
        _xy_deg = np.stack((_yaw_deg, _pitch_deg), axis=1)
        if _xy_deg.shape[0] != 0 and offset - onset > fixation_min_sample:  # if the entire interval is invalid, then we do NOT add it to fixation
            dispersion = np.max(_xy_deg, axis=0) - np.min(_xy_deg, axis=0)
            fixations.append(Fixation(duration, dispersion, saccades[i], saccades[i + 1], onset, offset, gaze_timestamps[onset], gaze_timestamps[offset], detection_alg=detection_alg))

    glitch_precentage = np.sum(events == -1) / len(events)

    for s in saccades:
        events[s.onset:s.offset] = SACCADE_CODE
    for f in fixations:
        events[f.onset:f.offset] = FIXATION_CODE
    print("Detected {} fixation from I-VT with {}% glitch percentage {}".format(len(fixations), glitch_precentage * 100, "" if head_rotation_xy_degree is None else "with head rotation"))
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


def gaze_event_detection_PatchSim(ps_fixation_detection_data, ps_fixation_detection_timestamps, events, long_gaze_threshold=0.15):
    fix_list_filtered = temporal_filter_fixation(ps_fixation_detection_data[1], marker_mode='marker', verbose=0)

    onset_timestamps = ps_fixation_detection_timestamps[fix_list_filtered == 1]
    offet_timestamps = ps_fixation_detection_timestamps[fix_list_filtered == 2]

    fixation_events = []

    fixated = []
    for onset_ts, offset_ts in zip(onset_timestamps, offet_timestamps):
        f = Fixation(offset_ts - onset_ts, None, None, None, None, None, onset_ts, offset_ts, "Patch-Sim")
        overlapping_gaze_intersects = get_overlapping_events(f.onset_time, f.offset_time, events, lambda x: type(x) == GazeRayIntersect)
        if len(overlapping_gaze_intersects) > 0:
            e = overlapping_gaze_intersects[0]  # IMPORTANT pick the first gaze event
            f = copy_item_info(f, e)
            if (f.item_index, f.block_id) not in fixated and f.duration > long_gaze_threshold:
                f.is_first_long_gaze = True
                fixated.append((f.item_index, f.block_id))
        else:
            f.dtn = dtnn_types['Null']
        fixation_events.append(f)
    print('Detected {} fixations from Patch similarity based fixation detection, {} are first long gaze on objects, {} targets, and {} distractors'.format(
        int(np.sum(fix_list_filtered[fix_list_filtered == 1])), len([f for f in fixation_events if f.is_first_long_gaze]),
        len([f for f in fixation_events if f.dtn==1]), len([f for f in fixation_events if f.dtn==2])))
    return fixation_events


class GazeRayIntersect(Event):
    def __init__(self, timestamp, onset_time, offset_time, is_first_long_gaze, *args, **kwargs):
        super().__init__(timestamp, *args, **kwargs)
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.is_first_long_gaze = is_first_long_gaze
