import numpy as np
import mne
import matplotlib.pyplot as plt
from mat4py import loadmat
import os

data_set_root = 'D:/HaowenWei/Data/HT_Data/fNIRS/FingerFootTapping'

assert os.path.exists(data_set_root), "File path does not exist."



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







file_names = [f for f in os.listdir(data_set_root) if os.path.isfile(os.path.join(data_set_root, f))]
file_names.sort()

data_dict = loadmat(os.path.join(data_set_root, file_names[0]))
fs = data_dict['nfo']['fs']
channel_names = data_dict['nfo']['clab']
channel_dict = {key: value for key, value in data_dict.items() if key.startswith('ch')}
data = list(channel_dict.values())
delta_HbO = np.array(data[0:20]).squeeze(axis=-1)
delta_HbR = np.array(data[20:40]).squeeze(axis=-1)
timestamps = np.arange(delta_HbO.shape[1])/fs
delta_HbO_channel_names = channel_names[0:20]
delta_HbR_channel_names = channel_names[20:40]
event_ts = np.array(data_dict['mrk']['time'])/1000
event = np.array(data_dict['mrk']['event']['desc'])
event_onehot = np.array(data_dict['mrk']['y']).T
class_names = data_dict['mrk']['className']

# delta_HbO_info = mne.create_info(ch_names=delta_HbO_channel_names, sfreq=fs, ch_types='hbo')
# delta_HbR_info = mne.create_info(ch_names=delta_HbR_channel_names, sfreq=fs, ch_types='hbr')

channel_types = ['hbo']*20 + ['hbr']*20
info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=channel_types)

raw = mne.io.RawArray(data=np.concatenate((delta_HbO, delta_HbR), axis=0), info=info, verbose=True)

print()


# plt.plot(data_dict['ch1'])
# plt.show()

# def process_finger_foot_tapping_mat():
#     pass

# if os.path.exists(data_set_root):
#     print("File path exists.")
# else:
#     print("File path does not exist.")

