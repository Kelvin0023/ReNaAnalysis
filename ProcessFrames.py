import cv2
import os
import pandas as pd

import numpy as np


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


image_folder = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/3-1-2022/0/ReNaUnityCameraCapture_03-01-2022-20-10-00'
gaze_info_file = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/3-1-2022/0/ReNaUnityCameraCapture_03-01-2022-20-10-00/GazeInfo.csv'
video_name = 'UnityFrames.avi'

gaze_info = pd.read_csv(gaze_info_file)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(x.strip('.png')))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

fovea_color = np.array([255, 0, 0], dtype=np.uint8)
parafovea_color = np.array([0, 255, 0], dtype=np.uint8)
peripheri_color = np.array([0, 0, 255], dtype=np.uint8)

bounding_box = (100, 100, 200, 200)

for i, image in enumerate(images):
    print('Processing {0} of {1} images'.format(i + 1, len(images)), end='\r', flush=True)
    img = cv2.imread(os.path.join(image_folder, image))

    gaze_coordinate = gaze_info.iloc[i, :].values
    gaze_x, gaze_y = int(gaze_coordinate[1]), int(gaze_coordinate[2])

    img = add_bounding_box(img, gaze_x, gaze_y, 10, 10, fovea_color)
    img = add_bounding_box(img, gaze_x, gaze_y, 25, 25, parafovea_color)
    img = add_bounding_box(img, gaze_x, gaze_y, 100, 70, peripheri_color)

    video.write(img)
    # if i == 200:
    #     break

cv2.destroyAllWindows()
video.release()