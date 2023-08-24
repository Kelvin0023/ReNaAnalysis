import json
import os
import pickle

import cv2
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

block_margin = 0.01
block_marker = 3
n_blocks = 20

include_dtn = False

cam_capture_folder = r'D:\Dropbox\Dropbox\ReNa\data\ReNaPilot-2022Fall\12_02_2022\ReNaUnityCameraCapture_12_02_2022_19_36_54'
file_path = r'D:\Dropbox\Dropbox\ReNa\data\ReNaPilot-2022Fall\12_02_2022\12_02_2022_19_39_19-Exp_RenaPilot-Sbj_zl-Ssn_2.p'
event_marker_path = 'renaanalysis/params/ReNaEventMarker.json'
eeg_preset_path = 'renaanalysis/params/BioSemi.json'
eyetracking_preset_path = 'renaanalysis/params/VarjoEyeDataComplete.json'

eeg_picks = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
data = pickle.load(open(file_path, 'rb'))

event_markers = json.load(open(event_marker_path, 'r'))
eeg_preset = json.load(open(eeg_preset_path, 'r'))
eyetracking_preset = json.load(open(eyetracking_preset_path, 'r'))


block_markers = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("BlockMarker"), :]
block_ids = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("BlockIDStartEnd"), :]
dtns = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("DTN"), :]
event_timestamps = data[event_markers['StreamName']][1]
condition_block_start_marker_indices = np.where(block_markers == block_marker)[0][:n_blocks]
condition_block_ids = block_ids[condition_block_start_marker_indices]
condition_block_end_marker_indices = [i for i in range(len(block_ids)) if block_ids[i] in -condition_block_ids][:n_blocks]

condition_block_start_times = event_timestamps[condition_block_start_marker_indices]
condition_block_end_times = event_timestamps[condition_block_end_marker_indices]


# workout the event marker stream #####################################################################################
if include_dtn:
    # create the dtn stream, m stands for marker
    dtn_stream = np.empty(0)
    dtn_timestamps = np.empty(0)
    last_timestamp = 0
    for m_start, m_end in zip(condition_block_start_marker_indices, condition_block_end_marker_indices):
        dtn_stream = np.concatenate((dtn_stream, dtns[m_start:m_end]))
        block_timestamps = event_timestamps[m_start:m_end] - event_timestamps[m_start] + last_timestamp + block_margin
        last_timestamp = block_timestamps[-1]
        dtn_timestamps = np.concatenate((dtn_timestamps, block_timestamps))


# workout the eeg stream #######################################################################################
montage = mne.channels.make_standard_montage('biosemi64')
pick_indices = [montage.ch_names.index(x) for x in eeg_picks]
eeg_timestamps = data['BioSemi'][1]
eeg_start_indices = [np.argmin(abs(eeg_timestamps - s)) for s in condition_block_start_times][:n_blocks]
eeg_end_indices = [np.argmin(abs(eeg_timestamps - s)) for s in condition_block_end_times][:n_blocks]
eeg_data = data['BioSemi'][0][1:65, :][np.array(pick_indices), :]

block_eeg_stream = np.empty((eeg_data.shape[0], 0))
block_eeg_timestamps = np.empty(0)
last_timestamp = 0

for eeg_start, eeg_end in zip(eeg_start_indices, eeg_end_indices):
    block_eeg_stream = np.concatenate((block_eeg_stream, eeg_data[:, eeg_start:eeg_end]), axis=1)
    block_timestamps = eeg_timestamps[eeg_start:eeg_end] - eeg_timestamps[eeg_start] + last_timestamp + block_margin
    last_timestamp = block_timestamps[-1]
    block_eeg_timestamps = np.concatenate((block_eeg_timestamps, block_timestamps))

block_eeg_stream = mne.filter.resample(block_eeg_stream, down=2 ** 4)
block_eeg_timestamps = block_eeg_timestamps[::2 ** 4]

# add the eyetracking frames ###################################################################################################
eye_timestamps = data['Unity.VarjoEyeTrackingComplete'][1]
eye_raw_timestamps = 1e-9 * data['Unity.VarjoEyeTrackingComplete'][0][eyetracking_preset['ChannelNames'].index('raw_timestamp')]
eye_timestamps = eye_raw_timestamps - eye_raw_timestamps[0] + eye_timestamps[0]  # use the raw timestamp to avoid the drift

eyetracking_start_indices = [np.argmin(abs(eye_timestamps - s)) for s in condition_block_start_times][:n_blocks]
eyetracking_end_indices = [np.argmin(abs(eye_timestamps - s)) for s in condition_block_end_times][:n_blocks]
eye_picks = [eyetracking_preset['ChannelNames'].index(f'gaze_forward_{ax}') for ax in ['x', 'y', 'z']] + [eyetracking_preset['ChannelNames'].index('status')]
eyetracking_data = data['Unity.VarjoEyeTrackingComplete'][0][eye_picks, :]

block_eyetracking_stream = np.empty((len(eye_picks), 0))
block_eyetracking_timestamps = np.empty(0)
last_timestamp = 0

for eye_start, eye_end in zip(eyetracking_start_indices, eyetracking_end_indices):
    block_eyetracking_stream = np.concatenate((block_eyetracking_stream, eyetracking_data[:, eye_start:eye_end]), axis=1)
    block_timestamps = eye_timestamps[eye_start:eye_end] - eye_timestamps[eye_start] + last_timestamp + block_margin
    last_timestamp = block_timestamps[-1]
    block_eyetracking_timestamps = np.concatenate((block_eyetracking_timestamps, block_timestamps))

# now the video stream ##############################################################################################
images = [img for img in os.listdir(cam_capture_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(x.strip('.png')))  # sort the image files
frame = cv2.imread(os.path.join(cam_capture_folder, images[0]))
dtype = frame.dtype

gaze_info_file = os.path.join(cam_capture_folder, 'GazeInfo.csv')
gaze_info = pd.read_csv(gaze_info_file)
image_timestamps = gaze_info['LocalClock'].to_numpy()
pixel_xs = gaze_info['GazePixelPositionX'].to_numpy()
pixel_ys = gaze_info['GazePixelPositionY'].to_numpy()

video_start_indices = [np.argmin(abs(image_timestamps - s)) for s in condition_block_start_times][:n_blocks]
video_end_indices = [np.argmin(abs(image_timestamps - s)) for s in condition_block_end_times][:n_blocks]

block_video_stream = np.empty((0, *frame.shape), dtype=dtype)
block_video_timestamps = np.empty(0)
block_pixel_stream = np.empty((0, 2))

last_timestamp = 0

for b_index, (start, end) in enumerate(zip(video_start_indices, video_end_indices)):
    print(f"loading frames for block {b_index} of {len(video_start_indices)}")
    block_frames = np.stack([cv2.imread(os.path.join(cam_capture_folder, f'{i}.png')) for i in range(start, end)])
    block_video_stream = np.concatenate((block_video_stream, block_frames), axis=0)
    block_pixel_stream = np.concatenate((block_pixel_stream, np.stack([pixel_xs[start:end], pixel_ys[start:end]]).T), axis=0)
    block_timestamps = image_timestamps[start:end] - image_timestamps[start] + last_timestamp + block_margin
    last_timestamp = block_timestamps[-1]
    block_video_timestamps = np.concatenate((block_video_timestamps, block_timestamps))

block_video_stream = np.rot90(block_video_stream, k=-1, axes=(1, 2))
block_video_stream = block_video_stream.reshape((len(block_video_stream), -1))
block_video_stream = block_video_stream.T
block_pixel_stream  = block_pixel_stream.T
# add another stream for video's gaze pixel location ################################################################





# save the data ############################################################################################################
out_data = {'Example-BioSemi-Midline': (block_eeg_stream, block_eeg_timestamps),
            'Example-Eyetracking': (block_eyetracking_stream, block_eyetracking_timestamps),
            'Example-Video-Gaze-Pixel': (block_pixel_stream, block_video_timestamps),
            'Example-Video': (block_video_stream, block_video_timestamps)}
if include_dtn:
    out_data['Example-EventMarker'] = (dtn_stream[None, :], dtn_timestamps)

pickle.dump(out_data, open(r'D:\PycharmProjects\RenaLabApp\examples/fixation-detection-example.p', 'wb'))