import mne.io
import mne_nirs
import mne_bids.stats
import numpy as np

print("fNIRS dataset example")
datapath = mne_nirs.datasets.fnirs_motor_group.data_path()
mne_bids.stats.count_events(datapath)

