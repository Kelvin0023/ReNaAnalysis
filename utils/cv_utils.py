import cv2
import os

# Read the video from specified path

def video_file_to_frame_array(path):
    # Path to video file
    video = cv2.VideoCapture(path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success = 1
    rtn = []
    while success:
        print("Extracting video frame {0}/{1}".format(count, length), end='\r')
        success, image = video.read()
        rtn.append(image)
        count += 1
    return rtn