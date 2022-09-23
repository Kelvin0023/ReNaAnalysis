import numpy as np
import mne
import pandas as pd

varjo_csv_header = ['raw_timestamp', 'log_time', 'focus_distance', 'frame_number', 'stability', 'status',
                    'gaze_forward_x', 'gaze_forward_y', 'gaze_forward_z', 'gaze_origin_x', 'gaze_origin_y',
                    'gaze_origin_z', 'HMD_position_x', 'HMD_position_y', 'HMD_position_z', 'HMD_rotation_x',
                    'HMD_rotation_y', 'HMD_rotation_z', 'left_forward_x', 'left_forward_y', 'left_forward_z',
                    'left_origin_x', 'left_origin_y', 'left_origin_z', 'left_pupil_size', 'left_status',
                    'right_forward_x', 'right_forward_y', 'right_forward_z', 'right_origin_x', 'right_origin_y',
                    'right_origin_z', 'right_pupil_size', 'right_status']


def varjo_epochs_to_df(epochs: mne.Epochs, resample=100):
    if resample: epochs_resampled = epochs.resample(resample)  # resample 100 Hz
    epochs_data = epochs_resampled.get_data()
    df_list = []
    for e in epochs_data:
        data = np.zeros([e.shape[-1], len(varjo_csv_header)])
        data[:, varjo_csv_header.index('raw_timestamp')] = e[0, :]
        data[:, varjo_csv_header.index('status')] = [round(s) for s in e[1, :]]
        data[:, varjo_csv_header.index('gaze_forward_x')] = e[2, :]
        data[:, varjo_csv_header.index('gaze_forward_y')] = e[3, :]
        df = pd.DataFrame(data=data, columns=varjo_csv_header)
        df_list.append(df)
    return df_list

def varjo_block_seq_to_df(block_sequence: np.ndarray):  # refer to add_event_markers_to_data_array, the first row is timestamp, the last four rows are event markers
    data =block_sequence[1: -4, :]
    df = pd.DataFrame(data=data.transpose(), columns=varjo_csv_header)
    df.reset_index()
    return df