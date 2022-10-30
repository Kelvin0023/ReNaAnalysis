from matplotlib import pyplot as plt


def visualiza_session(events):
    plt.rcParams["figure.figsize"] = [40, 3.5]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = [e.block_condition for e in events if e.is_block_start]

    dtn_timestamps = [e.timestamp for e in events if e.dtn]
    dtn_conditions = [e.block_condition for e in events if e.dtn]

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')
    (markers, stemlines, baseline) = plt.stem(dtn_timestamps, dtn_conditions, linefmt='orange', markerfmt='D', label='DTN conditions')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)
    plt.legend()
    plt.title('Condition Plot')
    plt.show()