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

image_folder = 'C:/Recordings/ReNaUnityCameraCapture_10-12-2022-02-19-52'
gaze_info_file = os.path.join(image_folder, 'GazeInfo.csv')
video_name = 'PatchComparison.mp4'

similarity_threshold = .02

fixation_y_value = -1e-2
fps = 30
video_fps = 30
video_start_frame = 0
video_frame_count = 5300  # this is 20 seconds
plot_time = video_frame_count - fps

patch_size = 63, 111  # width, height
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
previous_img_patch = None
distance_list = []
fixation_list = []
patch_boundaries = []
for i, image in enumerate(images[video_start_frame:]):  # iterate through the images
    print('Processing {0} of {1} images'.format(i + 1, video_frame_count), end='\r', flush=True)
    img = cv2.imread(os.path.join(image_folder, image))  # read in the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    img_modified = img.copy()
    gaze_coordinate = gaze_info.iloc[i, :].values  # get the gaze coordinate for this image
    gaze_x, gaze_y = int(gaze_coordinate[1]), int(gaze_coordinate[2])  # the gaze coordinate
    gaze_y = image_size[1] - gaze_y  # because CV's y zero is at the bottom of the screen
    center = gaze_x, gaze_y

    img_patch_x_min = int(np.min([np.max([0, gaze_x - patch_size[0] / 2]), image_size[0] - patch_size[0]]))
    img_patch_x_max = int(np.max([np.min([image_size[0], gaze_x + patch_size[0] / 2]), patch_size[0]]))
    img_patch_y_min = int(np.min([np.max([0, gaze_y - patch_size[1] / 2]), image_size[1] - patch_size[1]]))
    img_patch_y_max = int(np.max([np.min([image_size[1], gaze_y + patch_size[1] / 2]), patch_size[1]]))
    patch_boundaries.append((img_patch_x_min, img_patch_y_min, img_patch_x_max, img_patch_y_max))

    img_patch = img[img_patch_x_min: img_patch_x_max, img_patch_y_min: img_patch_y_max]

    # get similarity score
    if previous_img_patch is not None:
        img_tensor, previous_img_tensor = prepare_image_for_sim_score(img_patch), prepare_image_for_sim_score(previous_img_patch)
        distance = loss_fn_alex(img_tensor, previous_img_tensor).item()
        img_modified = cv2.putText(img_modified, "%.2f" % distance, center, cv2.FONT_HERSHEY_SIMPLEX, 1, center_color, 2, cv2.LINE_AA)
        distance_list.append(distance)
        fixation_list.append(0 if distance > similarity_threshold else 1)
    previous_img_patch = img_patch
    # if i % 10 == 0:
    #     plt.imshow(img_patch)
    #     plt.show()

    # draw the patch rectangle
    cv2.circle(img_modified, center, 1, center_color, 2)
    axis = (int(central_fov * ppds[0]), int(central_fov * ppds[1]))
    cv2.ellipse(img_modified, center, axis, 0, 0, 360, fovea_color, thickness=4)
    axis = (int(near_peripheral_fov * ppds[0]), int(near_peripheral_fov * ppds[1]))
    cv2.ellipse(img_modified, center, axis, 0, 0, 360, parafovea_color, thickness=4)
    axis = (int(1.25 * mid_perpheral_fov * ppds[0]), int(mid_perpheral_fov * ppds[1]))
    cv2.ellipse(img_modified, center, axis, 0, 0, 360, peripheri_color, thickness=4)

    images_with_bb.append(img_modified)
    # video.write(img)
    if i == video_frame_count:
        break

# duration thresholding
fixation_diff = np.diff(np.concatenate([[0], fixation_list]))
fix_onset_indices = np.argwhere(fixation_diff==1)
fix_offset_indices = np.argwhere(fixation_diff==-1)
fix_interval_indices = [(x[0], y[0]) for x, y in zip(fix_onset_indices, fix_offset_indices)]
fix_interval_indices = [x for x in fix_interval_indices if x[1] - x[0] > 5]  # only keep the fix interval longer than 150 ms == 5 frames
fix_list_filtered = np.empty(len(fixation_list))
fix_list_filtered[:] = np.nan
for index_onset, index_offset in fix_interval_indices:
    fix_list_filtered[index_onset:index_offset] = fixation_y_value # for visualization

for i, (fix, img, patch_boundary) in enumerate(zip(fix_list_filtered, images_with_bb, patch_boundaries)):
    img_modified = img.copy()
    if fix == fixation_y_value:
        shapes = np.zeros_like(img_modified, np.uint8)
        alpha = 0.3
        cv2.rectangle(shapes, patch_boundary[:2], patch_boundary[2:], patch_color, thickness=-1)
        mask = shapes.astype(bool)
        img_modified[mask] = cv2.addWeighted(img_modified, alpha, shapes, 1 - alpha, 0)[mask]
    else:
        cv2.rectangle(img_modified, patch_boundary[:2], patch_boundary[2:], patch_color, thickness=2)
    images_with_bb[i] = img_modified

clip = ImageSequenceClip.ImageSequenceClip(images_with_bb, fps=video_fps)
clip.write_videofile(video_name)


viz_time = 10

fig = plt.gcf()
fig.set_size_inches(30, 10.5)
plt.rcParams['font.size'] = '24'
plt.plot(np.linspace(0, viz_time, viz_time * fps), distance_list[:viz_time * fps], linewidth=5, label='Fovea Patch Distance')
plt.plot(np.linspace(0, viz_time, viz_time * fps), fix_list_filtered[:viz_time * fps], linewidth=10, label='Fixation')
plt.title('Example similarity distance sequence')
plt.ylabel("Similarity distance between previous and this frame")
plt.xlabel("Time (seconds)")
plt.show()