import math
from collections import defaultdict

import numpy as np
import scipy
from matplotlib import pyplot as plt

from renaanalysis.eye.EyeUtils import temporal_filter_fixation
from renaanalysis.params.params import dtnn_types, SACCADE_CODE, \
    FIXATION_CODE, varjoEyetracking_chs, headtracker_chs
from renaanalysis.utils.Event import Event, add_event_meta_info, get_events_between, is_event_in_block, copy_item_info, get_overlapping_events
from copy import copy

from renaanalysis.utils.interpolation import interpolate_array_nan
from renaanalysis.utils.jitter_removal import jitter_removal


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
        self.gaze_index = None  # index of this fixation on the object

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def consecutive_zeros_lengths(arr):
    # Find the indices where the array transitions from non-zero to zero
    transitions = np.where(np.diff(np.concatenate(([0], arr != 0, [0]))))[0]
    # Split the array based on the transitions and calculate the lengths
    lengths = np.diff(transitions)
    return lengths.tolist()

def fill_gap(gaze_xyz, gaze_timestamps, gaze_valid_array, max_gap_time=0.075, invalid_value=0):
    """
    notes that the invalid value must
    @param gaze_xyz:
    @param gaze_timestamps:
    @param gaze_valid_array:
    @param max_gap_time:
    @param invalid_value:
    @return:
    """
    gaze_xyz = np.copy(gaze_xyz)
    gaze_valid_array_valid_values = np.unique(gaze_valid_array)[np.unique(gaze_valid_array) != invalid_value]
    assert len(gaze_valid_array_valid_values) != 0, f'all values in gaze_valid_array are invalid: {invalid_value}'
    valid_value = gaze_valid_array_valid_values[0]

    valid_diff = np.diff(np.concatenate([[valid_value], gaze_valid_array, [valid_value]]))
    gap_start_indices = np.where(valid_diff < 0)[0]
    gap_end_indices = np.where(valid_diff > 0)[0]
    gaze_timestamps_extended = np.append(gaze_timestamps, gaze_timestamps[-1])

    interpolated_gap_count = 0
    interpolated_gap_durations = []
    ignored_gap_durations = []
    ignored_gap_start_end_indices = []

    for start, end in zip(gap_start_indices, gap_end_indices):
        if (gap_duration := gaze_timestamps_extended[end] - gaze_timestamps_extended[start]) > max_gap_time:
            ignored_gap_durations.append(gap_duration)
            ignored_gap_start_end_indices.append((start, end))
            continue
        else:
            interpolated_gap_count += 1
            gaze_xyz[:, start: end] = np.nan
            interpolated_gap_durations.append(gap_duration)
    plt.hist(interpolated_gap_durations + ignored_gap_durations, bins=100)
    plt.show()
    print(f"With max gap duration {max_gap_time * 1e3}ms, \n {interpolated_gap_count} gaps are interpolated among {len(gap_start_indices)} gaps, \n with interpolated gap with mean:median duration {np.mean(interpolated_gap_durations) *1e3}ms:{np.median(interpolated_gap_durations) *1e3}ms, \n and ignored gap with mean:median duration {np.mean(ignored_gap_durations) *1e3}ms:{np.median(ignored_gap_durations) *1e3}ms ")
    # interpolate the gaps
    gaze_xyz = interpolate_array_nan(gaze_xyz)

    # change the ignored gaps to nan
    for start, end in ignored_gap_start_end_indices:
        gaze_xyz[:, start: end] = np.nan
    return gaze_xyz

def _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps=None):
    eyetracking_data, eyetracking_lsl_timestamps = eyetracking_data_timestamps

    eyetracking_raw_timestamps = 1e-9 * eyetracking_data[varjoEyetracking_chs.index('raw_timestamp')]  # use the more accuracy raw timestamps for eyetracking
    eyetracking_timestamps = eyetracking_raw_timestamps + (eyetracking_lsl_timestamps[0] - eyetracking_raw_timestamps[0])

    assert eyetracking_data.shape[0] == len(varjoEyetracking_chs)
    gaze_xyz = np.copy(eyetracking_data[[varjoEyetracking_chs.index('gaze_forward_{0}'.format(x)) for x in ['x', 'y', 'z']]])
    status = eyetracking_data[varjoEyetracking_chs.index('status')]
    gaze_xyz = fill_gap(gaze_xyz, eyetracking_timestamps, status)

    # plt.plot(gaze_xyz[0, : 200], label='x')
    # plt.plot(gaze_xyz[1, : 200], label='y')
    # plt.plot(gaze_xyz[2, : 200], label='z')
    # plt.legend()
    # plt.show()

    head_rotation_xy_degree_eyesampled = None
    if headtracking_data_timestamps is not None:
        print("Processing headtracking data")
        headtracking_data, head_tracker_timestamps = headtracking_data_timestamps
        yaw_pitch_indices = headtracker_chs.index("Head Yaw"), headtracker_chs.index("Head Pitch")
        head_rotation_xy = headtracking_data[yaw_pitch_indices, :]
        # find the closest head tracking data point for each eyetracking point
        head_rotation_xy_degree_eyesampled = scipy.signal.resample(head_rotation_xy, len(eyetracking_timestamps),axis=1)

    return eyetracking_timestamps, gaze_xyz, status, head_rotation_xy_degree_eyesampled

def gaze_event_detection_I_DT(eyetracking_data_timestamps, events, headtracking_data_timestamps=None):
    eyetracking_raw_timestamps, gaze_xyz, gaze_status, head_rotation_xy_eyesampled = _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps)
    gaze_behavior_events, fixations, saccades, velocity = fixation_detection_i_dt(gaze_xyz,
                                                                                  gaze_timestamps=eyetracking_raw_timestamps,
                                                                                  # gaze_status=gaze_status,  no need to add gaze status as we are gap filling, which already accounts for gaze status
                                                                                  head_rotation_xy_degree=head_rotation_xy_eyesampled)
    fixations, saccades = add_event_info_to_gaze(fixations, events)
    print(', {} fixations are first long on objects. Targets: {}, Distractors {}'.format(len([f for f in fixations if f.is_first_long_gaze]), len([f for f in fixations if f.dtn==1]), len([f for f in fixations if f.dtn==2])))
    return fixations + saccades

def gaze_event_detection_I_VT(eyetracking_data_timestamps, events, headtracking_data_timestamps=None):
    eyetracking_raw_timestamps, gaze_xyz, gaze_status, head_rotation_xy_eyesampled = _preprocess_gaze_data(eyetracking_data_timestamps, headtracking_data_timestamps)

    gaze_behavior_events, fixations, saccades, velocity = fixation_detection_i_vt(gaze_xyz,
                                                                                  gaze_timestamps=eyetracking_raw_timestamps,
                                                                                  # gaze_status=gaze_status,  no need to add gaze status as we are gap filling, which already accounts for gaze status
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
    fixation_counts = defaultdict(int)  # (item_index, block_id) -> count
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

            if len(overlapping_gaze_intersects) > 0:  # if the fixation has overlapped a gaze intersect event
                fixation_counts[(f.item_index, f.block_id)] += 1
                f.gaze_index = fixation_counts[(f.item_index, f.block_id)] - 1
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
    """
    gaze vectors should be 3D vectors in the eye coordinate system, with the z axis pointing out of the eye straight ahead
    @param gaze_vector:
    @param head_rotation_xy_degree:
    @return:
    """

    # Calculate the angle of the 2D vector
    # gaze_yaw_degree = np.rad2deg(np.arctan2(gaze_vector[0, :], gaze_vector[2, :]))
    # gaze_pitch_degree = np.rad2deg(np.arctan2(gaze_vector[1, :], gaze_vector[2, :]))
    # gaze_angles_degree = np.rad2deg(np.arctan2(gaze_vector[1, :], gaze_vector[0, :]))

    # if head_rotation_xy_degree is not None:
        # head_rotation_xy_degree_diff = np.diff(head_rotation_xy_degree, prepend=head_rotation_xy_degree[:, 0][..., np.newaxis])
        # gaze_yaw_degree = gaze_yaw_degree + head_rotation_xy_degree_diff[0, :]
        # gaze_pitch_degree = gaze_pitch_degree + head_rotation_xy_degree_diff[1, :]
        #
        # head_yaw_rad = np.deg2rad(head_rotation_xy_degree[0, :])
        # head_pitch_rad = np.deg2rad(head_rotation_xy_degree[1, :])
        # x = np.sin(head_yaw_rad) * np.cos(head_pitch_rad)
        # y = np.sin(head_pitch_rad)
        # z = np.cos(head_yaw_rad) * np.cos(head_pitch_rad)
        # vector = np.array([x, y, z])
        #
        # gaze_angles_degree = gaze_angles_degree + np.rad2deg(np.arctan2(vector[1, :], vector[0, :]))
    # delta_gaze_angle_degree = np.diff(gaze_angles_degree, prepend=gaze_angles_degree[0])
    # return gaze_yaw_degree, gaze_pitch_degree, gaze_angles_degree

    # if head_rotation_xy_degree is not None:
    #     yaw_rad  = np.deg2rad(head_rotation_xy_degree[0, :])
    #     pitch_rad  = np.deg2rad(head_rotation_xy_degree[1, :])
    #
    #     head_x = np.cos(yaw_rad) * np.sin(pitch_rad)
    #     head_y = np.sin(yaw_rad) * np.sin(pitch_rad)
    #     head_z = np.cos(pitch_rad)
    #
    #     # add the head vector with the gaze vector
    #     gaze_vector = gaze_vector + np.stack([head_x, head_y, head_z])

    reference_vector = np.array([0, 0, 1])
    dot_products = np.dot(gaze_vector.T, reference_vector)
    magnitudes = np.linalg.norm(gaze_vector, axis=0)
    reference_magnitude = np.linalg.norm(reference_vector)
    cosine_angles = dot_products / (magnitudes * reference_magnitude)
    angles_rad = np.arccos(cosine_angles)
    angles_deg = np.degrees(angles_rad)

    return angles_deg


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

    gaze_angles_degree = _calculate_gaze_angles(gaze_xyz, head_rotation_xy_degree)

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
        # print(f"Running i-dt fixation detection, window start is at {gaze_point_window_start_index}, total length is {len(gaze_timestamps)}", end='\r')
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

def fixation_detection_i_vt(gaze_xyz, gaze_timestamps, gaze_status=None,
                            saccade_min_peak=6, saccade_min_amplitude=2, saccade_spacing=20e-3, saccade_min_sample=2,
                            fixation_min_sample=2, glitch_threshold=1000, head_rotation_xy_degree=None, fixation_min_duraiton_second=0.1):
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

    gaze_angles_degree = _calculate_gaze_angles(gaze_xyz, head_rotation_xy_degree)
    detection_alg = 'I-VT' if head_rotation_xy_degree is None else 'I-VT-Head'

    try:
        assert not np.any(np.diff(gaze_timestamps, prepend=1) == 0)
    except AssertionError:
        raise ValueError('Invalid gaze timestamps, time delta is zero at {0}'.format(np.argwhere(np.diff(gaze_timestamps, prepend=1) == 0)))
    velocities = np.gradient(gaze_angles_degree) / np.diff(gaze_timestamps, prepend=1)
    velocities[0] = velocities[1]  # assume the first velocity is the same as the first velocity
    speed = abs(velocities)
    events[velocities > glitch_threshold] = -1

    acceleration = np.gradient(velocities) / np.diff(gaze_timestamps, prepend=1)
    acceleration_sign_diff = np.concatenate([[0], np.diff(np.sign(acceleration))])

    acceleration_zero_crossings = np.where(np.logical_and(np.isfinite(acceleration), acceleration_sign_diff))[0] - 1

    # num_points = 100
    # zero_crossing_indices = [x for x in acceleration_zero_crossings if x < num_points]
    # plt.figure().set_figwidth(30)
    # plt.plot(velocities[:num_points], label='velocity')
    # plt.scatter(np.arange(0, num_points), velocities[:num_points])
    # plt.legend()
    # plt.twinx()
    # plt.plot(acceleration[:num_points], label='acceleration', color='orange')
    # plt.scatter(np.arange(0, num_points), acceleration[:num_points], color='orange')
    # plt.scatter(zero_crossing_indices, acceleration[zero_crossing_indices], color='red', label='a zero crossing')
    # plt.legend()
    # plt.twinx()
    # plt.show()
    # check if the first crossing is a onset, peak or an offset  # TODO complete checking acceleration zero crossings
    for crossing_i in range(0, len(acceleration_zero_crossings) - 2): # we start from the first onset, discard previous ones
        onset = acceleration_zero_crossings[crossing_i]
        peak = acceleration_zero_crossings[crossing_i + 1]
        offset = acceleration_zero_crossings[crossing_i + 2]

        # is_positive_peak = velocities[peak] > velocities[onset] and velocities[peak] > velocities[offset]
        # is_negative_peak = velocities[peak] < velocities[onset] and velocities[peak] < velocities[offset]
        # if not (is_positive_peak or is_negative_peak):  # check if the velocity peaks at peak

        is_peak = speed[peak] > speed[onset] and speed[peak] > speed[offset]
        if not is_peak:
            # check if the velocity peaks at peak
            continue
        if len(saccades) > 0 and gaze_timestamps[onset] - gaze_timestamps[saccades[-1].offset] < saccade_spacing:  # check temporal spacing condition for saccade
            continue
        if np.any(events[onset:offset] == -1):  # check if gaze status is invalid during the potential saccade
            continue
        if offset - onset < saccade_min_sample:
            continue
        if np.any(np.isnan(velocities[onset:offset])):
            continue

        amplitude = abs(gaze_angles_degree[offset] - gaze_angles_degree[onset])  # deg
        real_peak = onset + np.argmax(speed[onset:offset])  # peak may not be at the acceleration zero crossing depending on the sampling rate
        peak_speed = speed[real_peak]  # deg/s
        average_speed = np.mean(speed[onset:offset])  # deg/s
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]  # s
        if peak_speed > saccade_min_peak and amplitude > saccade_min_amplitude:
            saccades.append(
                Saccade(amplitude, duration, peak_speed, average_speed, onset, offset, gaze_timestamps[onset], gaze_timestamps[offset], real_peak, detection_alg=detection_alg))

    # num_points = 1000
    # zero_crossing_indices = [x for x in acceleration_zero_crossings if x < num_points]
    # plt.figure().set_figwidth(300)
    # plt.plot(velocities[:num_points], label='velocity')
    # plt.scatter(np.arange(0, num_points), velocities[:num_points])
    #
    # plotting_saccades = [saccade for saccade in saccades if saccade.offset < num_points]
    # plt.scatter([saccade.peak for saccade in plotting_saccades], [velocities[saccade.peak] for saccade in plotting_saccades], color='cyan', label='a saccade velocity peak', marker='X')
    # [plt.axvspan(saccade.onset, saccade.offset, facecolor='g', alpha=0.2) for saccade in plotting_saccades]
    #
    # plt.legend()
    # plt.twinx()
    # plt.plot(acceleration[:num_points], label='acceleration', color='orange')
    # plt.scatter(np.arange(0, num_points), acceleration[:num_points], color='orange')
    # plt.scatter(zero_crossing_indices, acceleration[zero_crossing_indices], color='red', label='a zero crossing')
    #
    # plt.legend()
    # plt.twinx()
    # plt.show()

    # identify the fixations for all the intervals between saccades
    fixation_interval_indices = [(saccades[i - 1].offset, saccades[i].onset) for i in range(1, len(saccades))]  # IGNORE the interval before the first saccade
    for i, (onset, offset) in enumerate(fixation_interval_indices):
        duration = gaze_timestamps[offset] - gaze_timestamps[onset]

        # _xy_deg = dtheta_degree[:, onset:offset][:, events[onset:offset] != -1] -    # check the dispersion excluding the invalid points
        # _yaw_deg = gaze_yaw_degree[onset:offset][events[onset:offset] != -1]   # check the dispersion excluding the invalid points
        # _pitch_deg = gaze_pitch_degree[onset:offset][events[onset:offset] != -1]   # check the dispersion excluding the invalid points
        # _xy_deg = np.stack((_yaw_deg, _pitch_deg), axis=1)
        fix_gaze_angles = gaze_angles_degree[onset:offset]
        if np.any(np.isnan(fix_gaze_angles)):  # don't detect for invalid intervals
            continue
        if gaze_timestamps[offset] - gaze_timestamps[onset] < fixation_min_duraiton_second:  # don't detect for short intervals
            continue
        if offset - onset > fixation_min_sample:  # if the entire interval is invalid, then we do NOT add it to fixation
            dispersion = np.max(fix_gaze_angles, axis=0) - np.min(fix_gaze_angles, axis=0)
            fixations.append(Fixation(duration, dispersion, saccades[i], saccades[i + 1], onset, offset, gaze_timestamps[onset], gaze_timestamps[offset], detection_alg=detection_alg))
    #
    # num_points = 1000
    # zero_crossing_indices = [x for x in acceleration_zero_crossings if x < num_points]
    # plt.figure().set_figwidth(300)
    # plt.plot(velocities[:num_points], label='velocity')
    # plt.scatter(np.arange(0, num_points), velocities[:num_points])
    #
    # plotting_saccades = [saccade for saccade in saccades if saccade.offset < num_points]
    # plt.scatter([saccade.peak for saccade in plotting_saccades], [velocities[saccade.peak] for saccade in plotting_saccades], color='cyan', label='a saccade velocity peak', marker='X')
    # [plt.axvspan(saccade.onset, saccade.offset, facecolor='g', alpha=0.2) for saccade in plotting_saccades]
    #
    # plotting_fixations = [fixation for fixation in fixations if fixation.offset < num_points]
    # [plt.axvspan(fixation.onset, fixation.offset, facecolor='r', alpha=0.2) for fixation in plotting_fixations]
    #
    # plt.legend()
    # plt.twinx()
    # plt.plot(acceleration[:num_points], label='acceleration', color='orange')
    # plt.scatter(np.arange(0, num_points), acceleration[:num_points], color='orange')
    # plt.scatter(zero_crossing_indices, acceleration[zero_crossing_indices], color='red', label='a zero crossing')
    #
    # plt.legend()
    # plt.twinx()
    # plt.show()

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
