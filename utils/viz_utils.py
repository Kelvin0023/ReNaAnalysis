import numpy as np
from matplotlib import pyplot as plt

from eye.eyetracking import Fixation, Saccade
from utils.Event import get_events_between, get_block_start_event, GazeRayIntersect
from params import *

def visualiza_session(events):
    plt.rcParams["figure.figsize"] = [40, 3.5]

    meta_block_timestamps = [e.timestamp for e in events if e.meta_block]
    meta_blocks = [e.meta_block for e in events if e.meta_block]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = [e.block_condition for e in events if e.is_block_start]

    dtn_timestamps = [e.timestamp for e in events if e.dtn]
    dtn_conditions = [e.block_condition for e in events if e.dtn]

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')
    (markers, stemlines, baseline) = plt.stem(dtn_timestamps, dtn_conditions, linefmt='orange', markerfmt='D', label='DTN conditions')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)

    (markers, stemlines, baseline) = plt.stem(meta_block_timestamps, meta_blocks, linefmt='cyan', markerfmt='D', label='meta blocks')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="cyan", markeredgewidth=2)

    plt.legend()
    plt.title('Session Conditions')
    plt.show()

def visualize_dtn(events, block_id):
    plt.rcParams["figure.figsize"] = [40, 5]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])

    plt.stem(block_start_timestamps, block_conditions, label='block start conditions')

    if block_id:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        # also plot the dtns
        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id==block_id])
        [plt.axvspan(onset, offset, alpha=0.5, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]
        plt.xlim(block_start_timestamp, block_end_timestamp)
    plt.legend()
    plt.show()


def visualize_gazeray(events, block_id=None):
    plt.rcParams["figure.figsize"] = [40, 5]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])
    gaze_intersects_dtn_timestamps = np.array([e.timestamp for e in events if e.gaze_intersect])
    gaze_intersects_dtn = np.array([e.dtn for e in events if e.gaze_intersect])

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')

    (markers, stemlines, baseline) = plt.stem(gaze_intersects_dtn_timestamps, gaze_intersects_dtn, label='gaze intersect DTN')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)

    if block_id:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        # also plot the dtns
        gaze_intersects_dtn_timestamps = np.array([e.timestamp for e in events if e.gaze_intersect])
        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id==block_id])
        [plt.axvspan(onset, offset, alpha=0.5, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]
        plt.xlim(block_start_timestamp, block_end_timestamp)
    plt.legend()
    plt.show()


def visualize_gaze_events(events, block_id=None, gaze_intersect_y=0.1, IDT_fix_y=.5, pathSim_fix_y = 1):
    f, ax = plt.subplots(figsize=[40, 5])

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])

    (markers, stemlines, baseline) = ax.stem(block_start_timestamps, block_conditions, label='block start conditions')

    # ax.scatter(gaze_intersects_dtn_timestamps, len(gaze_intersects_dtn_timestamps) * [gaze_intersect_y], marker='D', c=[dtn_color_dict[x] for x in gaze_intersects_dtn], label='gaze intersect DTN')

    if block_id:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id == block_id and e.dtn_onffset])
        [plt.axvspan(onset, offset, alpha=0.25, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]

        draw_fixations(ax, events, lambda x: type(x) == GazeRayIntersect and block_start_timestamp < x.timestamp < block_end_timestamp, gaze_intersect_y)
        draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'I-DT' and block_start_timestamp < x.timestamp < block_end_timestamp, IDT_fix_y)
        draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'Patch-Sim' and block_start_timestamp < x.timestamp < block_end_timestamp, pathSim_fix_y)

        ax.set_xlim(block_start_timestamp, block_end_timestamp)
        ax.set_title("Block ID {}, condition {}".format(block_id, get_block_start_event(block_id, events).block_condition))
    ax.legend()
    ax.set_xlabel('Time (sec)')
    plt.show()


def draw_fixations(ax, events, event_filter, fix_y):
    filtered_events = [e for e in events if event_filter(e)]
    fix_onset_times = [e.onset_time for e in filtered_events]
    fix_offset_times = [e.offset_time for e in filtered_events]
    fix_dtn = [e.dtn for e in filtered_events]
    for f_onset_ts, f_offset_ts, f_dtn in zip(fix_onset_times, fix_offset_times, fix_dtn):
        ax.hlines(y=fix_y, xmin=f_onset_ts, xmax=f_offset_ts, linewidth=4, colors=dtn_color_dict[f_dtn])