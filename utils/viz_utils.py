import numpy as np
from matplotlib import pyplot as plt


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

def visualize_dtn(events):
    plt.rcParams["figure.figsize"] = [40, 5]

    meta_block_timestamps = [e.timestamp for e in events if e.meta_block]
    meta_blocks = [e.meta_block for e in events if e.meta_block]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = [e.block_condition for e in events if e.is_block_start]

    dtn_timestamps = [e.timestamp for e in events if e.dtn and e.dtn_onffset]
    dtns = [e.dtn for e in events if e.dtn and e.dtn_onffset]

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')
    (markers, stemlines, baseline) = plt.stem(dtn_timestamps, dtns, linefmt='orange', markerfmt='D', label='Distractor/Target/Novelty')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)

    (markers, stemlines, baseline) = plt.stem(meta_block_timestamps, meta_blocks, linefmt='cyan', markerfmt='D', label='meta blocks')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="cyan", markeredgewidth=2)

    plt.legend()
    plt.title('Session DTN')
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

    plt.show()


def visualize_gaze_events(event, block_id=None):
    pass
    # start = 800
    # end = 1200
    # plt.rcParams["figure.figsize"] = (20, 10)
    # a = [s for s in saccades if s[0] > start and s[2] < end]
    # plt.plot(gaze_timestamps[start:end], velocities[start:end])
    # for s in a:
    #     plt.axvspan(gaze_timestamps[s[0]], gaze_timestamps[s[2]], alpha = 0.5, color='r')
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Velocity (deg/sec)')
    # plt.show()
    #
    # start = 800
    # end = 1200
    # plt.rcParams["figure.figsize"] = (20, 10)
    # a = [s for s in saccades if s[0] > start and s[2] < end]
    # b = [f for f in fixation_inteval_indices if f[0] > start and f[1] < end]
    # plt.plot(gaze_timestamps[start:end], velocities[start:end])
    # for s in a:
    #     plt.axvspan(gaze_timestamps[s[0]], gaze_timestamps[s[2]], alpha = 0.5, color='r')
    # for f in b:
    #     plt.axvspan(gaze_timestamps[f[0]], gaze_timestamps[f[1]], alpha = 0.5, color='g')
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Velocity (deg/sec)')
    # plt.show()

    # plt.hist([s[3].amplitude for s in saccades])
    # plt.show()
    # plt.hist([s[3].amplitude for s in saccades])
    # plt.show()
#    plt.plot(b_item_timestamps, b_gaze_ray_inter[1])
