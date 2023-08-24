import numpy as np
import mne
import matplotlib.pyplot as plt
from mat4py import loadmat
import os
import scipy
from sklearn.model_selection import train_test_split

fnirs_finger_and_foot_tapping_dataset_montage_channel_names = [
    'AFF5h', 'F5', 'F3', 'FFC5h', 'FC5', 'FC3', 'FCC5h', 'C5', 'C3', 'CCP5h',
    'AFF6h', 'F4','F6', 'FFC6h', 'FC4', 'FC6', 'FCC6h', 'C4', 'C6', 'CCP6h']


def create_montage(channel_names, standard_montage_name='standard_1005'):
    default_montage = mne.channels.make_standard_montage(standard_montage_name)
    default_montage_channel_positions = default_montage.get_positions()['ch_pos']
    new_channel_positions = {}
    for channel_name in channel_names:
        if channel_name in default_montage_channel_positions.keys():
            new_channel_positions[channel_name] = default_montage_channel_positions[channel_name]
        else:
            print(f"Channel {channel_name} not found in {standard_montage_name} montage.")
    new_montage = mne.channels.make_dig_montage(ch_pos=new_channel_positions, coord_frame='head')
    return new_montage

def fnirs_finger_and_foot_tapping_dataset_montage_channel_positions():
    montage = create_montage(fnirs_finger_and_foot_tapping_dataset_montage_channel_names)
    return list(montage.get_positions()['ch_pos'].values())*2


def visualize_epochs(epochs, event_groups, colors, picks, tmin_vis, tmax_vis, title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2),
                     is_plot_timeseries=True):
    """
    Visualize EEG epochs for different event types and channels.

    Args:
        epochs (mne.Epochs): The EEG epochs to visualize.
        event_groups (dict): A dictionary mapping event names to lists of event IDs. Only events in these groups will be plotted.
        colors (dict): A dictionary mapping event names to colors to use for plotting.
        picks (list): A list of EEG channels to plot.
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
        for ch in picks:
            for event_name, events in event_groups.items():
                try:
                    # Get the EEG data for the specified event type and channel
                    y = epochs.crop(tmin_vis, tmax_vis)[event_name].pick([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin_vis, tmax_vis, y.shape[-1])

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

def generate_mne_stim_channel(data_ts, event_ts, events, deviate=25e-2):
    """
    Generates an MNE stimulus channel from event markers and timestamps.

    Args:
        data_ts (np.ndarray): Timestamps for the data stream.
        event_ts (np.ndarray): Timestamps for event markers.
        events (np.ndarray): Event markers.
        deviate (float): Maximum acceptable jitter interval.

    Returns:
        array: MNE stimulus channel data.

    """
    stim_array = np.zeros((1, data_ts.shape[0]))
    events = np.reshape(events, (1, -1))
    event_data_indices = [np.argmin(np.abs(data_ts - t)) for t in event_ts if
                          np.min(np.abs(data_ts - t)) < deviate]

    for index, event_data_index in enumerate(event_data_indices):
        stim_array[0, event_data_index] = events[0, index]

    return stim_array


def add_stim_channel(raw_array, data_ts, event_ts, events, stim_channel_name='STI', deviate=25e-2):
    """
    Add a stimulation channel to the MNE raw data object.

    Args:
        raw_array (mne.io.RawArray): MNE raw data object.
        data_ts (numpy.ndarray): Timestamps for the data stream.
        event_ts (numpy.ndarray): Timestamps for event markers.
        events (numpy.ndarray): Event markers.
        stim_channel_name (str): Name of the stimulation channel. Default is 'STI'.
        deviate (float): Maximum acceptable jitter interval. Default is 0.25.

    Returns:
        None
    """
    stim_array = generate_mne_stim_channel(data_ts, event_ts, events, deviate=deviate)
    info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_array, info)
    raw_array.add_channels([stim_raw], force_update_info=True)



event_id = {
    'RIGHT': 1,
    'LEFT': 2,
    'FOOT': 3,
}

event_color = {
    'RIGHT': 'blue',
    'LEFT': 'red',
    'FOOT': 'yellow',
}




def get_fnirs_finger_and_foot_tapping_epoch_dict(dataset_root_dir, epoch_t_min, epoch_t_max, visualize_participants_epoch=True):
    assert os.path.exists(dataset_root_dir), "File path does not exist."
    file_names = [f for f in os.listdir(dataset_root_dir) if os.path.isfile(os.path.join(dataset_root_dir, f))]
    file_names.sort()

    participant_id = 0
    epoch_data_dict = {}
    for file_name in file_names:
        print('Processing file: {0}'.format(file_name))
        data_dict = loadmat(os.path.join(dataset_root_dir, file_name))
        fs = data_dict['nfo']['fs']
        # this_epoch_t_max = epoch_t_max - 0.5/fs
        channel_names = data_dict['nfo']['clab']
        channel_dict = {key: value for key, value in data_dict.items() if key.startswith('ch')}
        data = list(channel_dict.values())
        # delta_HbO = np.array(data[0:20]).squeeze(axis=-1)
        # delta_HbR = np.array(data[20:40]).squeeze(axis=-1)
        # timestamps = np.arange(delta_HbO.shape[1])/fs
        data = np.array(data).squeeze(axis=-1)
        timestamps = np.arange(data.shape[1]) / fs
        # delta_HbO_channel_names = channel_names[0:20]
        # delta_HbR_channel_names = channel_names[20:40]
        event_ts = np.array(data_dict['mrk']['time']) / 1000
        event = np.array(data_dict['mrk']['event']['desc'])
        event_onehot = np.array(data_dict['mrk']['y']).T
        class_names = data_dict['mrk']['className']

        # delta_HbO_info = mne.create_info(ch_names=delta_HbO_channel_names, sfreq=fs, ch_types='hbo')
        # delta_HbR_info = mne.create_info(ch_names=delta_HbR_channel_names, sfreq=fs, ch_types='hbr')

        channel_types = ['hbo'] * 20 + ['hbr'] * 20
        info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=channel_types)

        raw = mne.io.RawArray(data=data, info=info, verbose=True)
        add_stim_channel(raw, timestamps, event_ts, event, deviate=0.25)

        filtered_raw = raw.copy().filter(l_freq=0.01, h_freq=0.1, method='iir', verbose=True, picks=['hbo', 'hbr'])
        # resample raw
        spectrum_raw = raw.compute_psd()
        spectrum_raw.plot(average=True, picks="data", exclude="bads")

        spectrum_filtered_raw = filtered_raw.compute_psd()
        spectrum_filtered_raw.plot(average=True, picks="data", exclude="bads")

        event_groups = mne.find_events(raw, stim_channel='STI', verbose=True)
        raw_epoch = mne.Epochs(raw, event_groups, tmin=epoch_t_min, tmax=epoch_t_max, verbose=True, picks=['hbo', 'hbr'], event_id=event_id,
                               preload=True)
        filtered_raw_epoch = mne.Epochs(filtered_raw, event_groups, tmin=epoch_t_min, tmax=epoch_t_max, verbose=True, picks=['hbo', 'hbr'],
                                        event_id=event_id, preload=True)
        # resample epochs
        raw_epoch = raw_epoch.resample(20, npad='auto')
        filtered_raw_epoch = filtered_raw_epoch.resample(20, npad='auto')
        if visualize_participants_epoch:
            visualize_epochs(filtered_raw_epoch, event_id, event_color, picks=channel_names, tmin_vis=epoch_t_min, tmax_vis=epoch_t_max,
                             title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2),
                             is_plot_timeseries=True)

        epoch_data_dict[participant_id] = {'raw': [raw_epoch], 'filtered_raw': [filtered_raw_epoch]}
        participant_id += 1

        # # TODO: remove this
        # if participant_id > 1:
        #     break

    return epoch_data_dict



def get_fnirs_finger_and_foot_tapping_dataset(dataset_root_dir, epoch_t_min, epoch_t_max, visualize_participants_epoch=False):
    epoch_data_dict = get_fnirs_finger_and_foot_tapping_epoch_dict(dataset_root_dir=dataset_root_dir, epoch_t_min=epoch_t_min, epoch_t_max=epoch_t_max, visualize_participants_epoch=visualize_participants_epoch)

    x = []
    y = []

    subject_id = []
    run = []
    epoch_start_times = []
    channel_positions = []

    for this_subject_id, subject_data in epoch_data_dict.items():
        epochs_list = subject_data['filtered_raw']
        for run_index, epochs in enumerate(epochs_list):
            x.append(epochs.get_data())
            y.append(epochs.events[:, -1])
            subject_id.append([this_subject_id]*len(epochs))
            run.append([run_index]*len(epochs))
            epoch_start_times.append(epochs.events[:, 0]/epochs.info['sfreq'])
            channel_position = np.array(fnirs_finger_and_foot_tapping_dataset_montage_channel_positions())
            channel_positions.append(np.tile(channel_position, (len(epochs), 1, 1)))
            # channel_positions.append(np.array(fnirs_finger_and_foot_tapping_dataset_montage_channel_positions()))

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    subject_id = np.concatenate(subject_id, axis=0)
    run = np.concatenate(run, axis=0)
    epoch_start_times = np.concatenate(epoch_start_times, axis=0)
    channel_positions = np.concatenate(channel_positions, axis=0)

    metadata = {
        'subject_id': subject_id,
        'run': run,
        'epoch_start_times': epoch_start_times,
        'channel_positions': channel_positions
    }

    return x, y, metadata, event_color, epochs.info['sfreq']

#
# if __name__ == '__main__':
#     epoch_t_min = -1.5
#     epoch_t_max = 20
#     dataset_root_dir = 'D:/HaowenWei/Data/HT_Data/fNIRS/FingerFootTapping'
#
#     # epoch_data_dict = get_fnirs_finger_and_foot_tapping_epoch_dict(dataset_root_dir=dataset_root_dir, epoch_t_min=epoch_t_min, epoch_t_max=epoch_t_max)
#
#     x, y, metadata, event_color, fs = get_fnirs_finger_and_foot_tapping_dataset(dataset_root_dir=dataset_root_dir, epoch_t_min=epoch_t_min, epoch_t_max=epoch_t_max)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.1)
#     x_train = x_train.reshape(x_train.shape[0], -1)
#     x_test = x_test.reshape(x_test.shape[0], -1)
#
#     model.fit(x_train, y_train)

# if __name__ == '__main__':
#     create_montage(fnirs_finger_and_foot_tapping_dataset_montage_channel_names)
#
#


