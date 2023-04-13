import os
import os.path as op
import openneuro
import json
import numpy as np
import pandas as pd
import mne
import math
import matplotlib.pyplot as plt
import scipy

from mne.datasets import sample
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report,
                      find_matching_paths, get_entity_vals)


# colors = {
#     "response": "red",
#     "standard": "yellow",
#     "ignore": "blue",
#     "oddball": "orange",
#     "noise": "green",
#     "condition_5": "purple",
#     "noise_with_reponse": "pink",
#     "oddball_with_reponse": "black",
#     "standard_with_reponse": "gray"}

colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}

event_plot = {
    "standard": 1,
    "oddball_with_reponse": 7
}



def visualize_eeg_epochs(epochs, event_groups, colors, eeg_picks, title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2),
                         is_plot_timeseries=True):
    """
    Visualize EEG epochs for different event types and channels.

    Args:
        epochs (mne.Epochs): The EEG epochs to visualize.
        event_groups (dict): A dictionary mapping event names to lists of event IDs. Only events in these groups will be plotted.
        colors (dict): A dictionary mapping event names to colors to use for plotting.
        eeg_picks (list): A list of EEG channels to plot.
        title (str, optional): The title to use for the plot. Default is an empty string.
        out_dir (str, optional): The directory to save the plot to. If None, the plot will be displayed on screen. Default is None.
        verbose (str, optional): The verbosity level for MNE. Default is 'INFO'.
        fig_size (tuple, optional): The size of the figure in inches. Default is (12.8, 7.2).
        is_plot_timeseries (bool, optional): Whether to plot the EEG data as a timeseries. Default is True.

    Returns:
        None

    Raises:
        None

    """

    # Set the verbosity level for MNE
    mne.set_log_level(verbose=verbose)

    # Set the figure size for the plot
    plt.rcParams["figure.figsize"] = fig_size

    # Plot each EEG channel for each event type
    if is_plot_timeseries:
        for ch in eeg_picks:
            for event_name, events in event_groups.items():
                try:
                    # Get the EEG data for the specified event type and channel
                    y = epochs.crop(-0.1, 0.8)[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(-0.1, 0.8, y.shape[-1])

                # Plot the EEG data as a shaded area
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=colors[event_name], interpolate=True,
                                 alpha=0.5)
                plt.plot(time_vector, y_mean, c=colors[event_name], label='{0}, N={1}'.format(event_name, y.shape[0]))

            # Set the labels and title for the plot
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()
            plt.title('{0} - Channel {1}'.format(title, ch))

            # Save or show the plot
            if out_dir:
                plt.savefig(os.path.join(out_dir, '{0} - Channel {1}.png'.format(title, ch)))
                plt.clf()
            else:
                plt.show()


dataset = 'ds004022'
bids_root = 'E:/Data'
datatype = 'eeg'
task = 'P300'
num_subs = 13
num_runs = 3
subjects = {}
f = open(os.path.join(bids_root, f'task-{task}_events.json'))
events_info = json.load(f)
event_values = events_info['value']['Levels']
l = 0
for key in event_values:
    event_values[key] = l
    l += 1
for i in range(num_subs):
    subject = '{:0>{}}'.format(i + 1, 3)
    runs = {}
    subjects['sub-' + subject] = runs
# Download one subject's data from each dataset
# bids_root = op.join(op.dirname(sample.data_path()), dataset)
    extension = ".set"
    session = 'hc'
    bids_path = BIDSPath(root=bids_root, datatype=datatype)
    # bids_paths = find_matching_paths(bids_root, datatypes=datatype,
    #                                  sessions=sessions, extensions=extensions)

    suffix = 'eeg'
    for j in range(num_runs):
        bids_path = bids_path.update(subject=subject, task=task, suffix=suffix, run=str(j + 1), extension= extension)
        raw = read_raw_bids(bids_path=bids_path, verbose=True)

        tsv_path = f'E:/Data/sub-{subject}/eeg'

        event_file = open(os.path.join(tsv_path, f'sub-{subject}_task-{task}_run-{j+1}_events.tsv'))
        events = pd.read_csv(event_file, sep='\t')
        # sample_rate = 256
        # data = []
        event = np.zeros((events.shape[0], 3))
        my_event_dict = {}
        for k in range(events.shape[0]):
            # event_start = int(events['sample'][i] - 0.1 * sample_rate)
            # event_end = int(events['sample'][i] + 0.8 * sample_rate)
            # data.append(raw_data[:, event_start:event_end])

            event[k, 0] = events['sample'][k]
            event[k, 2] = event_values[events['value'][k]]
            if events['value'][k] not in my_event_dict:
                my_event_dict[events['value'][k]] = int(event[k, 2])
        # data = np.array(data)
        # type = np.array(type)
        # value = np.array(value)
        # event = np.array(event)
        event = event.astype(int)
        data = mne.Epochs(raw, event, event_id=my_event_dict, tmin=-0.1, tmax=0.8, baseline=(-0.1, 0), preload=True)
        runs['run-' + str(j + 1)] = data

visualize_eeg_epochs(subjects['sub-001']['run-1'], event_plot, colors, ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz',])
print(bids_path)
