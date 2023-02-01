import pickle

import numpy as np
from matplotlib import pyplot as plt

from renaanalysis.params.params import item_marker_names

data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Fall/Subjects/0/0.p'

data = pickle.load(open(data_path, 'rb'))

item_marker = data['Unity.ReNa.ItemMarkers'][0]
item_marker_timestamp = data['Unity.ReNa.ItemMarkers'][1]
item_marker_modified = np.copy(item_marker)


for i in range(item_marker_names.index('isGazeRayIntersected'), len(item_marker_modified), len(item_marker_names)):
    gaze_ray = np.copy(item_marker[i])
    gaze_ray_modified = np.clip(np.convolve(gaze_ray, 10 * [1.], mode='same'), 0, 1)
    # plt.plot(item_marker_timestamp[4000:6000], gaze_ray[4000:6000])
    # plt.show()
    item_marker_modified[i] = gaze_ray_modified

data['Unity.ReNa.ItemMarkers'][0] = item_marker_modified

pickle.dump(data, open(data_path, 'wb'))
