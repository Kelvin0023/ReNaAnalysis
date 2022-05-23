import math

import numpy as np
from matplotlib import pyplot as plt


class Saccade:
    def __init__(self, amplitude, duration, peak_velocity, average_velocity):
        self.amplitude = amplitude
        self.duration = duration
        self.peak_velocity = peak_velocity
        self.average_velocity = average_velocity


class Fixation:
    def __init__(self, duration, dispersion):
        self.duration = duration
        self.dispersion = dispersion


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def gaze_event_detection(gaze_xy, gaze_status, gaze_timestamps,
                         saccade_min_peak=6, saccade_min_amplitude=2, saccade_spacing=20e-3, glitch_threshold=1000):
    """
    gaze event detection based on Velocity Threshold Identification (I-VT)
    inspirations:
    https://journals.physiology.org/doi/abs/10.1152/jn.00237.2011
    https://dl.acm.org/doi/pdf/10.1145/1743666.1743682 for a comparison between different gaze event classification methods
    https://ieeexplore.ieee.org/abstract/document/8987791/ for some implementation of I-VT method
    currently no criterion is applied to the fixations

    @param saccade_min_peak: float: the peak velocity of a potential saccade period has to be greater than this minimal
    to be considered a saccade, unit is in deg/s
    @param saccade_min_amplitude: float: the minimal amplitude of for a saccade, unit is deg
    @param saccade_spacing: float: the minimal amplitude of for a saccade, unit is second, default to 20ms according to https://iovs.arvojournals.org/article.aspx?articleid=2193271
    @param glitch_threshold: float: unit in deg/s, time points with velocity exceeding this threshold will be considered
    a glitch in the recording
    @return
    event types: -1: noise or glitch; 1: fixation; 2: saccade
    """
    events = np.zeros(gaze_timestamps.shape)
    events[gaze_status != 2] = -1  # remove points where the status is invalid from the eyetracker
    saccades = []  # items are tuple of three: saccade onset index, saccade peak velocity index, saccade offset index, Saccade object,
    fixations = []  # items are tuple of three: saccade onset index, saccade peak velocity index, saccade offset index, Saccade object,

    gaze_xy_deg = (180 / math.pi) * np.arcsin(gaze_xy)

    # calculate eye velocity in degrees
    dxy = np.diff(gaze_xy_deg, axis=1, prepend=gaze_xy_deg[:, :1])
    dtheta = np.linalg.norm(dxy, axis=0)
    velocities = dtheta / np.diff(gaze_timestamps, prepend=1)
    velocities[0] = 0.  # assume the first velocity is 0

    events[velocities > glitch_threshold] = -1

    acceleration = np.diff(velocities, prepend=velocities[0])
    acceleration_zero_crossings = np.where(np.diff(np.sign(acceleration)))[0]
    for crossing_i in range(0, len(acceleration_zero_crossings) - 2):
        onset = acceleration_zero_crossings[crossing_i]
        peak = acceleration_zero_crossings[crossing_i + 1]
        offset = acceleration_zero_crossings[crossing_i + 2]

        if not (acceleration[peak] > 0 and acceleration[
            peak + 1] < 0):  # check the peak is cross from positive to negative
            continue
        if len(saccades) > 0 and gaze_timestamps[onset] - gaze_timestamps[
            saccades[-1][2]] < saccade_spacing:  # check temporal spacing condition for saccade
            continue
        if np.any(events[onset:offset] == -1):  # check if gaze status is invalid during the potential saccade
            continue

        amplitude = np.linalg.norm(gaze_xy_deg[:, offset] - gaze_xy_deg[:, onset], axis=0)
        peak_velocity = velocities[peak]
        average_velocity = np.mean(velocities[onset:offset])
        duration = offset - onset
        if velocities[peak] > saccade_min_peak and amplitude > saccade_min_amplitude:
            saccades.append((onset, peak, offset, Saccade(amplitude, duration, peak_velocity, average_velocity)))

    # identify the fixations for all the intervals between saccades
    fixation_inteval_indices = [(saccades[i-1][2], saccades[i][0]) for i in range(1, len(saccades))]  # IGNORE the interval before the first saccade
    for onset, offset in fixation_inteval_indices:
        _xy_deg = gaze_xy_deg[:, onset:offset][:, events[onset:offset] != -1] # check the dispersion excluding the invalid points
        dispersion = np.max(_xy_deg, axis=1) -  np.min(_xy_deg, axis=1)
        fixations.append(Fixation(offset - onset, dispersion))
    start = 800
    end = 1200
    plt.rcParams["figure.figsize"] = (20, 10)
    a = [s for s in saccades if s[0] > start and s[2] < end]
    plt.plot(gaze_timestamps[start:end], velocities[start:end])
    for s in a:
        plt.axvspan(gaze_timestamps[s[0]], gaze_timestamps[s[2]], alpha = 0.5, color='r')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (deg/sec)')
    plt.show()

    start = 800
    end = 1200
    plt.rcParams["figure.figsize"] = (20, 10)
    a = [s for s in saccades if s[0] > start and s[2] < end]
    b = [f for f in fixation_inteval_indices if f[0] > start and f[1] < end]
    plt.plot(gaze_timestamps[start:end], velocities[start:end])
    for s in a:
        plt.axvspan(gaze_timestamps[s[0]], gaze_timestamps[s[2]], alpha = 0.5, color='r')
    for f in b:
        plt.axvspan(gaze_timestamps[f[0]], gaze_timestamps[f[1]], alpha = 0.5, color='g')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (deg/sec)')
    plt.show()

    # plt.hist([s[3].amplitude for s in saccades])
    # plt.show()
    # plt.hist([s[3].amplitude for s in saccades])
    # plt.show()
    glitch_precentage = np.sum(events == -1) / len(events)
    print('')
