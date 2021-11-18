import os

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data_root = 'C:/Users/S-Vec/Downloads/drive-download-20211001T202725Z-008'
data_list = [sio.loadmat(os.path.join(data_root, fp)) for fp in os.listdir(data_root)]


eeg_timestamps = data['BioSemi'][0][1][0, :]
monitor1_timestamps = data['monitor1'][0][1][0, :]
unity_eventmarker_timestamps = data['Unity_NEDE_EventMarkers'][0][1][0, :]
eyetracking_timestamps = data['Unity_VarjoEyeTracking'][0][1][0, :]
eyetraccking_timestamps_diff = np.diff(eyetracking_timestamps)

# fig, ax = plt.subplots(tight_layout=True)
plt.hist(eyetraccking_timestamps_diff,
         bins=np.linspace(4.9e-3, 5.1e-3, 1000))
plt.xlabel('Varjo eyetracking sampling interval')
plt.ylabel('Frequency')
plt.show()

np.mean(eyetraccking_timestamps_diff)
np.std(eyetraccking_timestamps_diff)

plt.plot(eyetracking_timestamps, label="Varjo Timestamps")
plt.plot(eeg_timestamps, label="BioSemi (EEG) Timestamps")
plt.plot(monitor1_timestamps, label="Screen capture Timestamps")
plt.plot(unity_eventmarker_timestamps, label="Unity Eventmarker Timestamps")
plt.xlabel("Timestamp indices")
plt.ylabel("Timestamp value (sec)")
plt.legend()
plt.show()

len(eyetracking_timestamps) / (eeg_timestamps[-1] - eeg_timestamps[0])