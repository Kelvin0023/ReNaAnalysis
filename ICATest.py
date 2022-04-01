import mne
from rena.utils.data_utils import RNStream
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
plt.ion()

data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects/0/0.dats'
data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)

eeg_data = data['BioSemi'][0][:, :2048 * 20]
eeg_data = eeg_data * 10e-6  # convert to uV

biosemi_64_montage = mne.channels.make_standard_montage('biosemi64')
data_channel_names = biosemi_64_montage.ch_names

info = mne.create_info(
    ['Trig1'] + data_channel_names + ['EX{0}'.format(x) for x in range(1, 9)] + ['AUX{0}'.format(x) for x in range(1, 17)],
    sfreq=2048,
    ch_types=['misc'] + ['eeg'] * len(data_channel_names) + ['ecg'] * 2 + 22 * ['misc'])
raw = mne.io.RawArray(eeg_data, info)
raw.set_montage(biosemi_64_montage)
raw, _ = mne.set_eeg_reference(raw, 'average',
                               projection=False)
raw = raw.copy().filter(l_freq=1, h_freq=50)  # bandpass filter
raw = raw.copy().notch_filter(freqs=np.arange(60, 241, 60), filter_length='auto')

regexp = r'(Fp.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
         show_scrollbars=False, scalings=dict(eeg=400e-6))
