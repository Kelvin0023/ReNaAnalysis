import math

import pandas as pd
import numpy as np

from renaanalysis.eye.eyetracking import gaze_event_detection
from renaanalysis.utils.utils import interpolate_array_nan

data_path = '/Users/Leo/Desktop/02_26_23_gaze_positions_on_surface_Surface 1.csv'
# data_path = '/Users/Leo/Desktop/gaze_positions.csv'

data = pd.read_csv(data_path)
gaze_xy = data[['gaze_normal1_x', 'gaze_normal1_y']].values
timestamps = data['gaze_timestamp'].values
gaze_xy = interpolate_array_nan(gaze_xy.T)  # interpolate missing values (possibly blinks)

sr = len(timestamps) / (timestamps[-1] - timestamps[0])
1 / np.mean(np.diff(timestamps))
# gaze_xy_deg = (180 / math.pi) * np.arcsin(gaze_xy)

gaze_behavior_events, fixations, saccades, velocity = gaze_event_detection(gaze_xy, gaze_timestamps=timestamps)
