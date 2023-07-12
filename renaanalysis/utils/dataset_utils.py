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
from RenaAnalysis import get_rdf

from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report,
                      find_matching_paths, get_entity_vals)
from sklearn.preprocessing import LabelEncoder

from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.learning.preprocess import preprocess_samples_eeg_pupil, preprocess_samples_and_save
from renaanalysis.params.params import eeg_name, pupil_name
from renaanalysis.utils.Bidict import Bidict
from renaanalysis.utils.data_utils import epochs_to_class_samples
from renaanalysis.utils.multimodal import PhysioArray, MultiModalArrays
from renaanalysis.utils.rdf_utils import rena_epochs_to_class_samples_rdf
from renaanalysis.utils.utils import preprocess_standard_eeg


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

class DataSet():
    def __init__(self, hierarchy_list, epochs):
        self.orged_epochs = hierarchy_list

    def organize_epochs(self, hierarchy_list, epochs):
        for hierarchy_key in hierarchy_list:
            pass




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

            raw = preprocess_standard_eeg(raw, ica_path=os.path.join(os.path.dirname(bids_path), 'sub-' + subject + '_task-' + task + '_run-' + str(j + 1) + '_ica.fif'),
                                          is_running_ica=False,
                                          ocular_artifact_mode='proxy', blink_ica_threshold=np.linspace(10, 7, 5), eyemovement_ica_threshold=np.linspace(2.5, 2.0, 5))

            tsv_path = os.path.join(bids_root, f'sub-{subject}/{suffix}')
            epoch_info_tsv = open(os.path.join(tsv_path, f'sub-{subject}_task-{task}_run-{j + 1}_events.tsv'))
            epochs_info = pd.read_csv(epoch_info_tsv, sep='\t')

            eventID_mat = np.zeros((epochs_info.shape[0], 3))
            event_dict_this_run = {}
            metadata_dict = {'subject_id': [], 'run': []}
            for k in range(epochs_info.shape[0]):
                eventID_mat[k, 0] = epochs_info['sample'][k]
                eventID_mat[k, 2] = event_label_dict[epochs_info['value'][k]]
                if epochs_info['value'][k] not in event_dict_this_run:
                    event_dict_this_run[epochs_info['value'][k]] = int(eventID_mat[k, 2])
                metadata_dict['subject_id'].append(subject)
                metadata_dict['run'].append(str(j))
            eventID_mat = eventID_mat.astype(int)
            metadata = pd.DataFrame(metadata_dict)
            data = mne.Epochs(raw, eventID_mat, event_id=event_dict_this_run, metadata=metadata, tmin=epoch_tmin, tmax=epoch_tmax, baseline=baseline_tuple,
                              preload=True)
            runs['run-' + str(j + 1)] = data
    return subjects

def load_auditory_oddball_data(bids_root, srate=256, epoch_tmin = -0.1, epoch_tmax = 0.8, include_last=False, colors=None):
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

def parse_file_tree(directory):
    result = {}
    for entry in os.scandir(directory):
        if entry.is_file():
            result[entry.name] = None
        elif entry.is_dir():
            result[entry.name] = parse_file_tree(entry.path)
    return result

def get_BCI_montage(montage_name, picks=None):
    '''
    This function returns the standard BCI montages with the specified picks

    @param montage_name: The name of the montage, e.g. 'standard_1005'
    @param picks: The list of channels to be kept
    @return: montage: The montage object
    '''
    montage = mne.channels.make_standard_montage(montage_name)
    if picks is not None:
        ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in picks]
        montage.ch_names = [montage.ch_names[x] for x in ind]
        kept_channel_info = [montage.dig[x + 3] for x in ind]
        # Keep the first three rows as they are the fiducial points information
        montage.dig = montage.dig[0:3] + kept_channel_info
    return montage

def get_BCICIV_samples(data_root, event_viz_colors, eeg_resample_rate=200, ):
    '''
    This function returns the samples of the BCICIV dataset

    @param data_root: The root directory of the dataset
    @return:
    '''
    file_tree_dict = parse_file_tree(data_root)
    kept_channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1',
                     'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    channel_mapping = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'}
    mont1020 = get_BCI_montage('standard_1020', picks=kept_channels)
    event_id_mapping = {'769': 0, '770': 1, '771': 2, '772': 3, '276': 4, '277': 5, '768': 6, '783': 7, '1023': 8, '1072': 9, '32766': 10}
    subjects = []
    for file_name, _ in file_tree_dict.items():
        if 'T' in file_name:
            raw = mne.io.read_raw_gdf(os.path.join(data_root, file_name), preload=True)
            mne.rename_channels(raw.info, channel_mapping)
            raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right']) # otherwise the channel names are not consistent with montage
            events, event_id = mne.events_from_annotations(raw, event_id=event_id_mapping)
            is_merge_event = 'drop'
            data = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.8, baseline=(-0.1, 0), preload=True, event_repeated=is_merge_event)
            data.set_montage(mont1020)
            subjects.append(data)
    all_epochs = mne.concatenate_epochs(subjects)
    x, y, start_time, metadata = epochs_to_class_samples(all_epochs, list(event_viz_colors.keys()), picks=kept_channels,
                                                         reject='auto', n_jobs=16,
                                                         eeg_resample_rate=eeg_resample_rate, colors=event_viz_colors)
    return x, y, start_time, metadata


def get_DEAP_preprocessed_samples(data_root):
    ratings = pd.read_csv(os.path.join(data_root, 'metadata_csv/participant_ratings.csv'))

    data_directory = os.path.join(data_root, 'data_preprocessed_python')
    file_tree_dict = parse_file_tree(data_directory)
    kept_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    mont1020 = mne.channels.make_standard_montage('standard_1020', picks=kept_channels)
    subjects_data = []
    subjects_label = []
    for file_name, _ in file_tree_dict.items():
        data = pickle.load(open(os.path.join(data_directory, file_name), 'rb'), encoding='latin1')
        subjects_data.append(data['data'])
        subjects_label.append(data['labels'])
    x = np.concatenate(subjects_data, axis=0)
    y = np.concatenate(subjects_label, axis=0)
    subject_id = ratings['Participant_id'].values
    sorted_ratings = ratings.sort_values(['Participant_id', 'Experiment_id'])
    start_time = sorted_ratings['Start_time'].values
    return x, y, subject_id, start_time, mont1020

def get_DEAP_samples(data_root, event_names=None, picks=None, event_viz_colors=None, eeg_resample_rate=200, subject_picks=None):

    # Read metadata participant rating
    ratings = pd.read_csv(os.path.join(data_root,'metadata_csv/participant_ratings.csv'))

    # Specify the root directory of the file tree
    data_directory = os.path.join(data_root, 'data_original')

    # Parse the file tree and obtain the dictionary representation
    file_tree_dict = parse_file_tree(data_directory)
    idx = 1
    kept_channels = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    mont1020 = mne.channels.make_standard_montage('standard_1020', picks=kept_channels)
    subjects = []
    event_name_map = {'HighValanceHighArousal': 0, 'HighValanceLowArousal': 1, 'LowValanceHighArousal': 2, 'LowValanceLowArousal': 3}
    for file_name, _ in file_tree_dict.items():
        metadata_dict = {'subject_id': [], }
        raw = mne.io.read_raw_bdf(os.path.join(data_directory, file_name),preload=True)
        # raw = preprocess_standard_eeg(raw, ica_path=os.path.join(os.path.dirname(data_root),
        #                                                          'sub-' + str(idx) + '_ica.fif'),
        #                               is_running_ica=False,
        #                               ocular_artifact_mode='proxy', blink_ica_threshold=np.linspace(10, 7, 5),
        #                               eyemovement_ica_threshold=np.linspace(2.5, 2.0, 5))
        subject_ratings = ratings[ratings['Participant_id'] == idx]
        eventID_mat = np.zeros((len(subject_ratings), 3), dtype=int)
        event_dict_this_run = {}
        for k in range(len(subject_ratings)):
            eventID_mat[k, 0] = int(subject_ratings['Start_time'][k + (idx-1)*40]/1e6*raw.info['sfreq'])
            eventID_mat[k, 2] = get_DEAP_epoch_label(subject_ratings, k + (idx-1)*40)
            if eventID_mat[k, 2] == 0:
                event_dict_this_run['HighValanceHighArousal'] = 0
            elif eventID_mat[k, 2] == 1:
                event_dict_this_run['HighValanceLowArousal'] = 1
            elif eventID_mat[k, 2] == 2:
                event_dict_this_run['LowValanceHighArousal'] = 2
            else:
                event_dict_this_run['LowValanceLowArousal'] = 3
            metadata_dict['subject_id'].append(idx)
        metadata = pd.DataFrame(metadata_dict)
        idx += 1
        data = mne.Epochs(raw, eventID_mat, event_id=event_dict_this_run, metadata=metadata, tmin=-0.1, tmax=5, preload=True, baseline=(-0.1, 0))
        subjects.append(data)
    all_epochs = mne.concatenate_epochs(subjects)
    x, y, start_time, metadata = epochs_to_class_samples(all_epochs, list(event_viz_colors.keys()), picks=picks,
                                                         reject='auto', n_jobs=16,
                                                         eeg_resample_rate=eeg_resample_rate, colors=event_viz_colors)

def get_DEAP_epoch_label(ratings_data, k):
    if ratings_data['Valence'][k] >= 5 and ratings_data['Arousal'][k] >= 5:
        return 0
    elif ratings_data['Valence'][k] >= 5 and ratings_data['Arousal'][k] < 5:
        return 1
    elif ratings_data['Valence'][k] < 5 and ratings_data['Arousal'][k] >= 5:
        return 2
    else:
        return 3


def get_TUHG_samples(data_root, export_data_root, epoch_length, event_names, picks, colors, eeg_resample_rate, subject_picks=None):

    # Specify the root directory of the file tree
    root_directory = data_root

    # Parse the file tree and obtain the dictionary representation
    file_tree_dict = parse_file_tree(root_directory)
    subjects = {}
    for subject_group_id, subject_group in file_tree_dict.items():
        subjects[subject_group_id] = {}
        for subject_name, subject in subject_group.items():
            subjects[subject_group_id][subject_name] = {}
            if subject_picks is None:
                for session_name, session in subject.items():
                    subjects[subject_group_id][subject_name][session_name] = {}
                    for montage_type_name, montage_type in session.items():
                        subjects[subject_group_id][subject_name][session_name][montage_type_name] = {}
                        for data_file_name, _ in montage_type.items():
                            metadata_dict = {'subject_group_id': [], 'subject_name': [], 'session_name': [], 'montage_type_name': []}
                            raw = mne.io.read_raw_edf(os.path.join(data_root,
                                                                   f'{subject_group_id}/{subject_name}/{session_name}/{montage_type_name}/{data_file_name}'),
                                                      preload=True)
                            num_epochs = math.ceil(raw.__len__()/(epoch_length*raw.info['sfreq']))
                            eventID_mat = np.zeros((num_epochs, 3))
                            for k in range(num_epochs):
                                eventID_mat[k, 0] = k * epoch_length
                                eventID_mat[k, 2] = 0
                                metadata_dict['subject_group_id'].append(subject_group_id)
                                metadata_dict['subject_name'].append(subject_name)
                                metadata_dict['session_name'].append(session_name)
                                metadata_dict['montage_type_name'].append(montage_type_name)
                            metadata = pd.DataFrame(metadata_dict)
                            data = mne.Epochs(raw, eventID_mat, {0: 'standard'}, metadata=metadata, tmin=0, tmax=epoch_length*raw.info['sfreq'], preload=True)
                            subjects[subject_group_id][subject_name][session_name][montage_type_name][data_file_name] = data
            elif subject_name in subject_picks:
                for session_name, session in subject.items():
                    subjects[subject_group_id][subject_name][session_name] = {}
                    for montage_type_name, montage_type in session.items():
                        subjects[subject_group_id][subject_name][session_name][montage_type_name] = {}
                        for data_file_name, _ in montage_type.items():
                            metadata_dict = {'subject_group_id': [], 'subject_name': [], 'session_name': [],
                                             'montage_type_name': []}
                            raw = mne.io.read_raw_edf(os.path.join(data_root,
                                                                   f'{subject_group_id}/{subject_name}/{session_name}/{montage_type_name}/{data_file_name}'),
                                                      preload=True)
                            num_epochs = math.ceil(raw.__len__() / (epoch_length * raw.info['sfreq']))
                            eventID_mat = np.zeros((num_epochs, 3), dtype='int')
                            for k in range(num_epochs):
                                eventID_mat[k, 0] = int(k * epoch_length * raw.info['sfreq'])
                                eventID_mat[k, 2] = 0
                                metadata_dict['subject_group_id'].append(subject_group_id)
                                metadata_dict['subject_name'].append(subject_name)
                                metadata_dict['session_name'].append(session_name)
                                metadata_dict['montage_type_name'].append(montage_type_name)
                            metadata = pd.DataFrame(metadata_dict)
                            data = mne.Epochs(raw, eventID_mat, {'standard': 0}, metadata=metadata, tmin=0,
                                              tmax=epoch_length, preload=True, baseline=(0, 0))
                            subjects[subject_group_id][subject_name][session_name][montage_type_name][data_file_name] = data
    # pickle.dump(subjects, open(export_data_root, 'wb'))
    all_epochs = []
    for subject_group_id, subject_group in subjects.items():
        for subject_name, subject in subject_group.items():
            for session_name, session in subject.items():
                for montage_type_name, montage_type in session.items():
                    for _, data in montage_type.items():
                        all_epochs.append(data)
    all_epochs = mne.concatenate_epochs(all_epochs)
    x, y, start_time, metadata = epochs_to_class_samples(all_epochs, event_names, picks=picks, reject='auto', n_jobs=16,
                                                         eeg_resample_rate=eeg_resample_rate, colors=colors)
    return x, y, start_time, metadata

def get_auditory_oddball_samples(bids_root, export_data_root, is_regenerate_epochs, reject, eeg_resample_rate, picks='eeg'):
    event_viz_colors = {
        "standard": "red",
        "oddball_with_reponse": "green"
    }
    if is_regenerate_epochs:
        subjects = load_auditory_oddball_data(bids_root=bids_root, colors=event_viz_colors)
        all_epochs = []
        for subject_key, run_values in subjects.items():
            for run_key, run in run_values.items():
                all_epochs.append(run)
        all_epochs = mne.concatenate_epochs(all_epochs)
        x, y, start_time, metadata = epochs_to_class_samples(all_epochs, list(event_viz_colors.keys()), picks=picks, reject=reject, n_jobs=16,
                                       eeg_resample_rate=eeg_resample_rate, colors=event_viz_colors)

        pickle.dump(x, open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'wb'))
        pickle.dump(y, open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'wb'))
        pickle.dump(start_time, open(os.path.join(export_data_root, 'start_time_auditory_oddball.p'), 'wb'))
        pickle.dump(metadata, open(os.path.join(export_data_root, 'metadata_auditory_oddball.p'), 'wb'))
    else:
        x = pickle.load(open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'rb'))
        y = pickle.load(open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'rb'))
        start_time = pickle.load(open(os.path.join(export_data_root, 'start_time_auditory_oddball.p'), 'rb'))
        metadata = pickle.load(open(os.path.join(export_data_root, 'metadata_auditory_oddball.p'), 'rb'))
    # le = LabelEncoder()
    # Y_encoded = le.fit_transform(y)

    print(f"Load data took {time.time() - start_time} seconds")
    return x, y, start_time, metadata, event_viz_colors



def get_rena_samples(base_root, export_data_root, is_regenerate_epochs, reject, exg_resample_rate, eyetracking_resample_srate, locking_name='VS-I-VT-Head', participant=None, session=None):
    """

    @param base_root:
    @param export_data_root:
    @param is_regenerate_epochs:
    @param event_names:
    @param reject:
    @param exg_resample_rate:
    @param colors:
    @param picks:
    @param locking_name: locking_name can be any keys in locking_name_filters
    @return:
    """
    conditions = Bidict({'RSVP': 1., 'Carousel': 2., 'VS': 3., 'TS': 4., 'TSgnd': 8, 'TSid': 9})
    dtnn_types = Bidict({'Distractor': 1, 'Target': 2, 'Novelty': 3, 'Null': 4})
    event_viz_colors = {'Distractor': 'blue', 'Target': 'red'}

    locking_name_filters_dict = {
                            'VS-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                            'VS-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]],
                            'VS-I-DT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]],
                            'VS-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                     lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]],
                            'RSVP-Item-Onset': [lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                                lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Target"]],

                            'Carousel-Item-Onset': [lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                                    lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Target"]],

                            'RSVP-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                            'RSVP-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP']  and x.dtn==dtnn_types["Target"]],
                            'RSVP-I-DT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
                                    lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]],
                            'RSVP-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                     lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]],

                            'Carousel-I-VT-Head': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Distractor"],
                                                    lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Target"]],
                            'Carousel-FLGI': [lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Distractor"],
                                            lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Target"]],
                            'Carousel-I-DT-Head': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-DT-Head' and x.dtn == dtnn_types["Distractor"],
                                            lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-DT-Head' and x.dtn == dtnn_types["Target"]],
                            'Carousel-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                                    lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]
                                        } #nyamu <3
    if is_regenerate_epochs:
        rdf = get_rdf(base_root=base_root, exg_resample_rate=exg_resample_rate, ocular_artifact_mode='proxy')
        pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))
        x, y, _, _ = rena_epochs_to_class_samples_rdf(rdf, event_names, locking_name_filters_dict[locking_name], data_type='both', rebalance=False, participant=participant, session=session, plots='full', exg_resample_rate=exg_resample_rate, eyetracking_resample_srate=eyetracking_resample_srate, reject=reject)
        pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
        pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'wb'))

    else:
        # rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
        try:
            x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
            y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
        except FileNotFoundError:
            raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}, locking {locking_name}" + ", EEGPupil" if model_name == 'EEGPupil' else "")
    return x, y, event_viz_colors


def get_dataset(dataset_name, epochs_root=None, data_root=None, is_regenerate_epochs=False, reject='auto',
                eeg_resample_rate=200, is_apply_pca_ica_eeg=True, pca_ica_eeg_n_components=20,
                eyetracking_resample_srate=20):
    """

    @param is_regenerate_epochs: whether to regenerate epochs or not, if set to False, the function will attempt
    to read original data from data_root. If set to True, the function will attempt to read epochs from epochs_root.
    The latter is usually faster because it loads preprocessed epochs directly.
    It is also recommended to use an SSD for both data_root and epochs_root because it is much faster than HDD.
    @param dataset_name: can be 'auditory_oddball', 'rena', TODO 'TUH', 'DEAP', and more

    @return:
    """
    if not is_regenerate_epochs:
        assert data_root is not None, "data_root must be specified if is_regenerate_epochs is False"
    else:
        assert epochs_root is not None, "epochs_root must be specified if is_regenerate_epochs is True"

    if dataset_name == 'auditory_oddball':

        x, y, start_time, metadata, event_viz_colors = get_auditory_oddball_samples(data_root, epochs_root, is_regenerate_epochs, reject, eeg_resample_rate)
        physio_arrays = [PhysioArray(x, sampling_rate=eeg_resample_rate, physio_type=eeg_name, dataset_name=dataset_name)]
    elif dataset_name == "rena":
        x, y, event_viz_colors = get_rena_samples(data_root, epochs_root, is_regenerate_epochs, reject, eeg_resample_rate, eyetracking_resample_srate)
        physio_arrays = [PhysioArray(x[0], sampling_rate=eeg_resample_rate, physio_type=eeg_name, dataset_name=dataset_name),
              PhysioArray(x[1], sampling_rate=eyetracking_resample_srate, physio_type=pupil_name, dataset_name=dataset_name)]
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    physio_arrays = preprocess_samples_and_save(physio_arrays, epochs_root, is_apply_pca_ica_eeg, pca_ica_eeg_n_components)

    return MultiModalArrays(physio_arrays, labels_array=y, dataset_name=dataset_name, event_viz_colors=event_viz_colors)