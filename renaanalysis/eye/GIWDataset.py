import os.path

import cv2
import numpy as np
from scipy import io
from utils.cv_utils import video_file_to_frame_array

# loaded = io.loadmat('D:/FinalSet_GIW/Extracted_Data/Extracted_Data/Ball_Catch/Labels/PrIdx_1_TrIdx_2_Lbr_1.mat', squeeze_me=True, struct_as_record=False)
# data = loaded['LabelData'][0]

data_path = "D:/FinalSet_GIW/Extracted_Data/Extracted_Data/Ball_Catch/ProcessData_cleaned/PrIdx_1_TrIdx_2.mat"
scene_video_path = "D:/FinalSet_GIW/1/2/world.mp4"  # TODO this video path should be got from the PrIdx and TrIdx
output_path = "D:/Dropbox/Dropbox/ReNa/FinalSet_GIW/out/1_2_ballcatch"

loaded = io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
data = loaded['ProcessData'].ETG
participant_index = loaded['ProcessData'].PrIdx  #
trial_index = loaded['ProcessData'].TrIdx  #
trial_index = loaded['ProcessData'].TrIdx  #
# scene_res = np.flip(data.SceneResolution)

scene_frame_nums = data.SceneFrameNo
unique_scene_frame_nums = np.unique(scene_frame_nums)
true_labels = data.Labels
scene_res = data.SceneResolution
PORs = data.POR
# PORs[:, 1] = 1 - PORs[:, 1]
PORs_pixel = np.stack([PORs[:, 0] * scene_res[0], PORs[:, 1] * scene_res[1]], axis=1)
PORs_pixel = np.flip(PORs_pixel, axis=1)
video_frames = video_file_to_frame_array(scene_video_path)

# iterate through the scene frames and save the video frames
# for i, frame_num in enumerate(unique_scene_frame_nums):
#     print("Outputting {}/{} frames".format(i, len(unique_scene_frame_nums)), end='\r')
#     data_i = np.argwhere(scene_frame_nums == frame_num)[0, 0]  # get the first index with the matching frame number; there will be a few entries with match frame number because the eye-tracker's sampling rate is much higher than that of the scene camera
#     cv2.imwrite(os.path.join(output_path, '{}.png'.format(i)), video_frames[frame_num])

# save the gaze info
gaze_info = []
for i, frame_num in enumerate(unique_scene_frame_nums):
    data_i = np.argwhere(scene_frame_nums == frame_num)[0, 0]  # get the first index with the matching frame number; there will be a few entries with match frame number because the eye-tracker's sampling rate is much higher than that of the scene camera
    # gaze_info.append([i] + np.flip(PORs_pixel[data_i]).tolist() + [true_labels[i]])
    gaze_info.append([i] + np.flip(PORs_pixel[data_i]).tolist() + [true_labels[i]])
gaze_info = np.array(gaze_info)
np.savetxt(os.path.join(output_path, "GazeInfo.csv"), gaze_info, delimiter=",")