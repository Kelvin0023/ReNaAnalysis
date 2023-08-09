import mne.io
import mne_nirs
import mne_bids.stats
import numpy as np

print("fNIRS dataset example")

data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

data_array = mne.io.RawArray(data, info)
