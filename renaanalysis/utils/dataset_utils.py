import os
import os.path as op
import time

import openneuro
import json
import numpy as np
import pandas as pd
import mne
import math
import matplotlib.pyplot as plt
import scipy
import pickle

from mne.datasets import sample
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report,
                      find_matching_paths, get_entity_vals)
from sklearn.preprocessing import LabelEncoder

from renaanalysis.utils.data_utils import epochs_to_class_samples


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


def load_epoched_data_tsv_event_info(num_subs, num_runs, bids_root, subject_id_width, datatype, task, suffix, extension, event_label_dict, epoch_tmin, epoch_tmax, baseline_tuple):
    """
        Load epoched data from 3rd party datasets from provided

        Args:
            num_subs : number of subjects.
            num_runs : number of runs.
            bids_root : the root path of the dataset.
            subject_id_width : width of the subject representation, if the subject representation is '001', then subject_id_width = 3.
            datatype : datatype of the data. Can be 'eeg', 'meg' ...
            task : task name of the data.
            suffix : suffix of the data.
            extension : extension name of the data.
            event_label_dict : a dictionary with event label as key and a number id as value.
            epoch_tmin : the start time of the epoch with respect to the event marker, in seconds.
            epoch_tmax : the end time of the epoch with respect to the event marker, in seconds.
            baseline_tuple : a tuple of the baseline start time and end time. For example, (-0.1, 0)

        Returns:
            A dictionary containing all the subjects, which then contains each run of epoched data

        Raises:
            None

        """
    subjects = {}
    for i in range(num_subs):
        subject = '{:0>{}}'.format(i + 1, subject_id_width)
        runs = {}
        subjects['sub-' + subject] = runs
        bids_path = BIDSPath(root=bids_root, datatype=datatype)
        for j in range(num_runs):
            bids_path = bids_path.update(subject=subject, task=task, suffix=suffix, run=str(j + 1), extension=extension)
            raw = read_raw_bids(bids_path=bids_path, verbose=True)
            tsv_path = os.path.join(bids_root, f'sub-{subject}/{suffix}')
            epoch_info_tsv = open(os.path.join(tsv_path, f'sub-{subject}_task-{task}_run-{j + 1}_events.tsv'))
            epochs_info = pd.read_csv(epoch_info_tsv, sep='\t')

            eventID_mat = np.zeros((epochs_info.shape[0], 3))
            event_dict_this_run = {}
            for k in range(epochs_info.shape[0]):
                eventID_mat[k, 0] = epochs_info['sample'][k]
                eventID_mat[k, 2] = event_label_dict[epochs_info['value'][k]]
                if epochs_info['value'][k] not in event_dict_this_run:
                    event_dict_this_run[epochs_info['value'][k]] = int(eventID_mat[k, 2])
            eventID_mat = eventID_mat.astype(int)
            data = mne.Epochs(raw, eventID_mat, event_id=event_dict_this_run, tmin=epoch_tmin, tmax=epoch_tmax, baseline=baseline_tuple,
                              preload=True)
            runs['run-' + str(j + 1)] = data
    return subjects

def load_auditory_oddball_data(bids_root, srate=256, epoch_tmin = -0.1, epoch_tmax = 0.8, include_last=False):
    colors = {
        "standard": "red",
        "oddball_with_reponse": "green"
    }

    event_plot = {
        "standard": 1,
        "oddball_with_reponse": 7
    }
    datatype = 'eeg'
    task = 'P300'
    extension = ".set"
    suffix = 'eeg'
    num_subs = 13
    num_runs = 3
    subject_id_width = 3
    f = open(os.path.join(bids_root, f'task-{task}_events.json'))
    events_info = json.load(f)
    event_label_dict = events_info['value']['Levels']

    event_label_dict = dict([(key.replace('_', ' ') if 'condition' in key else key, value) for key, value in event_label_dict.items()])
    l = 0
    for key in event_label_dict:
        event_label_dict[key] = l
        l += 1



    if not include_last:
        epoch_tmax -= 1/srate

    baseline_tuple = (-0.1, 0)
    subjects = load_epoched_data_tsv_event_info(num_subs, num_runs, bids_root, subject_id_width, datatype, task, suffix, extension, event_label_dict, epoch_tmin, epoch_tmax, baseline_tuple)

    visualize_eeg_epochs(subjects['sub-001']['run-1'], event_plot, colors, ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz'])
    # pickle.dump(subjects, open(os.path.join('./3rd_party_data/audio_oddball', f'subjects.p'), 'wb'))
    return subjects

def get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, eeg_resample_rate, colors):
    start_time = time.time()  # record the start time of the analysis
    if not reload_saved_samples:
        subjects = load_auditory_oddball_data(bids_root=bids_root)
        all_epochs = []
        for subject_key, run_values in subjects.items():
            for run_key, run in run_values.items():
                all_epochs.append(run)
        all_epochs = mne.concatenate_epochs(all_epochs)
        x, y = epochs_to_class_samples(all_epochs, event_names, picks=picks, reject=reject, n_jobs=16,
                                       eeg_resample_rate=eeg_resample_rate, colors=colors)

        pickle.dump(x, open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'wb'))
        pickle.dump(y, open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'wb'))
    else:
        x = pickle.load(open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'rb'))
        y = pickle.load(open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'rb'))

    le = LabelEncoder()
    Y_encoded = le.fit_transform(y)

    print(f"Load data took {time.time() - start_time} seconds")
    return x, Y_encoded, le