import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import numpy as np

fp = '/home/apocalyvec/Dropbox/research/NiDyn/s1/s1_run1_Eye.xdf'


data, header = pyxdf.load_xdf(fp)

eyetracking_timestamps = data[5]['time_stamps']
eyetraccking_timestamps_diff = np.diff(eyetracking_timestamps)

plt.hist(eyetraccking_timestamps_diff,
         bins=np.linspace(eyetraccking_timestamps_diff.min(), eyetraccking_timestamps_diff.max(), 100))
plt.xlabel('Vive (previous implementation) eyetracking sampling interval')
plt.ylabel('Frequency')
plt.show()
