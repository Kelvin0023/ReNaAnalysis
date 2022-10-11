import cv2
import os

import lpips
import moviepy
import pandas as pd

import numpy as np
from PIL import ImageDraw
from moviepy.video.io import ImageSequenceClip
import matplotlib.pyplot as plt
from eye.EyeUtils import prepare_image_for_sim_score


def add_bounding_box(a, x, y, width, height, color):
    copy = np.copy(a)
    image_height = a.shape[0]
    image_width = a.shape[1]
    bounding_box = (np.max([0, x - int(width/2)]), np.max([0, y - int(height/2)]), width, height)

    copy[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    copy[np.min([image_height-1, bounding_box[1] + bounding_box[3]]), bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], np.min([image_width-1, bounding_box[0] + bounding_box[2]])] = color
    return copy

image_folder = 'C:/Users/Lab-User/Dropbox/ReNa/data/ReNaPilot-2022Spring/3-5-2022/0/ReNaUnityCameraCapture_03-05-2022-13-42-32'
gaze_info_file = os.path.join(image_folder, 'GazeInfo.csv')
video_name = 'PatchComparison.mp4'

similarity_threshold = .1

plot_time = 20
fps = 30
video_start_frame = 2789
video_frame_count = plot_time * fps  # this is 20 seconds
patch_size = 200, 200  # width, height
fovs = 115, 90  # horizontal, vertical, in degrees
central_fov = 13  # fov of the fovea
near_peripheral_fov = 30  # physio fov
mid_perpheral_fov = 60  # physio fov

# for drawing the fovea box
patch_color = (255, 255, 0)
center_color = (255, 0, 0)
fovea_color = (0, 255, 0)
parafovea_color = (0, 0, 255)
peripheri_color = (0, 255, 255)
# END OF USER PARAMETERS ###############################################################

# read the gaze info csv
gaze_info = pd.read_csv(gaze_info_file)

# get the video frames
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
image_size = cv2.imread(os.path.join(image_folder, images[0])).shape[:2]
ppds = image_size[0] / fovs[0], image_size[1] / fovs[1]  # horizontal, vertical, calculate pixel per degree of FOV

images.sort(key=lambda x: int(x.strip('.png')))  # sort the image files
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
images_with_bb = []
previous_img = None
distance = None
distance_list = []
fixation_list = []
for i, image in enumerate(images[video_start_frame:]):  # iterate through the images
    print('Processing {0} of {1} images'.format(i + 1, video_frame_count), end='\r', flush=True)
    img = cv2.imread(os.path.join(image_folder, image))  # read in the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    gaze_coordinate = gaze_info.iloc[i, :].values  # get the gaze coordinate for this image
    gaze_x, gaze_y = int(gaze_coordinate[1]), int(gaze_coordinate[2])  # the gaze coordinate
    gaze_y = image_size[1] - gaze_y  # because CV's y zero is at the bottom of the screen
    center = gaze_x, gaze_y

    if previous_img is not None:
        img_tensor, previous_img_tensor = prepare_image_for_sim_score(img), prepare_image_for_sim_score(previous_img)
        distance = loss_fn_alex(img_tensor, previous_img_tensor).item()
        distance_list.append(distance)
        fixation_list.append(0 if distance > similarity_threshold else 1)

    previous_img = img
    if distance is not None:
        img = cv2.putText(img, "%.2f" % distance, center, cv2.FONT_HERSHEY_SIMPLEX, 1, center_color, 2, cv2.LINE_AA)

    img = add_bounding_box(img, gaze_x, gaze_y, 63, 111, patch_color)
    cv2.circle(img, center, 1, center_color, 2)
    axis = (int(central_fov * ppds[0]), int(central_fov * ppds[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, fovea_color, thickness=4)
    axis = (int(near_peripheral_fov * ppds[0]), int(near_peripheral_fov * ppds[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, parafovea_color, thickness=4)
    axis = (int(1.25 * mid_perpheral_fov * ppds[0]), int(mid_perpheral_fov * ppds[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, peripheri_color, thickness=4)

    images_with_bb.append(img)
    # video.write(img)
    if i == video_frame_count:
        break


clip = ImageSequenceClip.ImageSequenceClip(images_with_bb, fps=fps)
clip.write_videofile(video_name)


fixation_diff = np.diff(np.concatenate([[0], fixation_list]))
fix_onset_indices = np.argwhere(fixation_diff==1)
fix_offset_indices = np.argwhere(fixation_diff==-1)
fix_interval_indices = [(x[0], y[0]) for x, y in zip(fix_onset_indices, fix_offset_indices)]
fix_interval_indices = [x for x in fix_interval_indices if x[1] - x[0] > 5]  # only keep the fix interval longer than 150 ms == 5 frames
fix_list_filtered = np.empty(len(fixation_list))
fix_list_filtered[:] = np.nan
for index_onset, index_offset in fix_interval_indices:
    fix_list_filtered[index_onset:index_offset] = 0.0  # for visualization

viz_time = 10

fig = plt.gcf()
fig.set_size_inches(30, 10.5)
plt.rcParams['font.size'] = '24'
plt.plot(np.linspace(0, viz_time, viz_time * fps), distance_list[:viz_time * fps], linewidth=5, label='Fovea Patch Distance')
plt.plot(np.linspace(0, viz_time, viz_time * fps), fix_list_filtered[:viz_time * fps], linewidth=10, label='Fixation')
plt.title('Example similarity score sequence')
plt.xlabel("Similarity distance between previous and this frame")
plt.ylabel("Time (seconds)")
plt.show()