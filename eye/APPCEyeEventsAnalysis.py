import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#%%
from eye.eyetracking import gaze_event_detection, plot_gaze_events_overlay

data_path = 'C:/Users/Lab-User/Downloads/APPC_1.csv'

screen_diagnal = 76.2  # in cm, for 30 inches screen, assuming this is the measure of the actual screen space
screen_size = 2560, 1600  # width and height
srate = 500
fixation_heat_radius = 5  # in degree field of view, use 5 degrees for the central view

width_centimeter = math.sin(math.atan(screen_size[0]/screen_size[1])) * screen_diagnal
centimeter_per_pixel = width_centimeter / screen_size[0]  # used to convert the gaze xy from pixel to degrees

df = pd.read_csv(data_path)
gaze_xy = np.array([df['x'].values, df['y'].values])
gaze_xy_deg = gaze_xy * centimeter_per_pixel
gaze_timestamps = np.linspace(0, gaze_xy.shape[1] / srate, gaze_xy.shape[1])  # create the approximate timestamps based on the sampling rate

gaze_behavior_events, fixations, saccades, velocities = gaze_event_detection(gaze_xy, gaze_timestamps, gaze_xy_format='angle')

# convert the fixation map into headmap
fixation_onset_points = [f.onset for f in fixations]


# plot a sample gaze sequence
plot_gaze_events_overlay(1700, 1705, gaze_timestamps, saccades, fixations, velocities)