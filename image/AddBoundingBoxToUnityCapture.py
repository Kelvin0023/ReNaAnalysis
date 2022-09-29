import cv2
import os

import moviepy
import pandas as pd

import numpy as np
from PIL import ImageDraw
from moviepy.video.io import ImageSequenceClip


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


image_folder = 'C:/Users/Lab-User/Dropbox/ReNa/ReNaData/ReNaPilot-2022Spring/3-1-2022/0/ReNaUnityCameraCapture_03-01-2022-20-10-00'
gaze_info_file = 'C:/Users/Lab-User/Dropbox/ReNa/ReNaData/ReNaPilot-2022Spring/3-1-2022/0/ReNaUnityCameraCapture_03-01-2022-20-10-00/GazeInfo.csv'
video_name = 'UnityFrames.mp4'
fps = 30
fovs = 115, 90
central_fov = 13
near_peripheral_fov = 30
mid_perpheral_fov = 60

gaze_info = pd.read_csv(gaze_info_file)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
image_size = cv2.imread(os.path.join(image_folder, images[0])).shape[:2]
ppis = image_size[0] / fovs[0], image_size[1] / fovs[1]

images.sort(key=lambda x: int(x.strip('.png')))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

center_color = (255, 0, 0)
fovea_color = (0, 255, 0)
parafovea_color = (0, 0, 255)
peripheri_color = (0, 255, 255)

bounding_box = (100, 100, 200, 200)

images_with_bb = []
for i, image in enumerate(images):
    print('Processing {0} of {1} images'.format(i + 1, len(images)), end='\r', flush=True)
    img = cv2.imread(os.path.join(image_folder, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gaze_coordinate = gaze_info.iloc[i, :].values
    gaze_x, gaze_y = int(gaze_coordinate[1]), int(gaze_coordinate[2])
    gaze_y = image_size[1] - gaze_y
    # img = add_bounding_box(img, gaze_x, gaze_y, 10, 10, fovea_color)
    # img = add_bounding_box(img, gaze_x, gaze_y, 25, 25, parafovea_color)
    # img = add_bounding_box(img, gaze_x, gaze_y, 100, 70, peripheri_color)
    center = gaze_x, gaze_y
    cv2.circle(img, center, 1, center_color, 2)

    axis = (int(central_fov * ppis[0]), int(central_fov * ppis[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, fovea_color, thickness=4)
    axis = (int(near_peripheral_fov * ppis[0]), int(near_peripheral_fov * ppis[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, parafovea_color, thickness=4)
    axis = (int(1.25 * mid_perpheral_fov * ppis[0]), int(mid_perpheral_fov * ppis[1]))
    cv2.ellipse(img, center, axis, 0, 0, 360, peripheri_color, thickness=4)

    images_with_bb.append(img)
    # video.write(img)
    if i == 1500:
        break

clip = ImageSequenceClip.ImageSequenceClip(images_with_bb, fps=fps)
clip.write_videofile(video_name)