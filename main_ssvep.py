import matplotlib.pyplot as plt
import mne
import numpy as np
from mne import Epochs
from mne.time_frequency import psd_welch

from renaanalysis.params.params import eeg_montage, eeg_channel_names

data_path = 'trial1.bdf'
event_ids = {'15Hz': 15,  '35Hz': 35, '10Hz': 10, '30Hz': 30, '20Hz': 20, '25Hz': 25, '5Hz': 5}
eeg_picks = ['PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
n_jobs = 16

raw = mne.io.read_raw_bdf(data_path, preload=True)
raw_data = raw.get_data()

# update the stim channel
stim_from_raw = np.copy(raw_data[-1, :]) - 65279
stim_from_raw[stim_from_raw == 2 ** 16] = 0
stim_from_raw = np.diff(np.concatenate([stim_from_raw, [0]]))

_stim = np.zeros(len(stim_from_raw))
_stim[stim_from_raw == 2 ** 8] = list(event_ids.values())
raw_data[-1, :] = _stim

data_channels = eeg_channel_names + ['stim']
data_channel_types = ['eeg'] * len(eeg_channel_names) + ['stim']

info = mne.create_info(data_channels, sfreq=2048, ch_types=data_channel_types)  # with 3 additional info markers and design matrix
raw = mne.io.RawArray(raw_data, info)

raw = raw.set_montage(eeg_montage, match_case=False)
raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)

raw = raw.filter(l_freq=1, h_freq=50, n_jobs=n_jobs, picks='eeg')  # bandpass filter for brain
raw = raw.resample(256, n_jobs=n_jobs)

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw, picks='eeg')
ica.plot_sources(raw)
ica.plot_components()
ica_excludes = input("Enter manual ICA components to exclude (use space to deliminate): ")
if len(ica_excludes) > 0:
    ica.exclude += [int(x) for x in ica_excludes.split(' ')]
print(f'Excluding ica components: {ica.exclude}')
ica.apply(raw)

found_events = mne.find_events(raw, stim_channel='stim')
epochs = Epochs(raw, events=found_events, event_id=event_ids, tmin=-0.5, tmax=10.0, baseline=(-0.5, 0.0), preload=True, verbose=False, picks='eeg')

epochs['35Hz'].pick('POz').plot_psd(fmax=40, tmin=2, tmax=6)

for event_name, event_id in event_ids.items():
    psd, freq = psd_welch(epochs[event_name].pick(eeg_picks), n_fft=1028, n_per_seg=256 * 3, picks='all')
    psd = 10 * np.log10(psd)
    psd_mean = np.mean(psd[0], axis=0)

    plt.plot(freq, psd_mean, color='b', label='Channel Average')
    plt.title(f'{eeg_picks} for event {event_name}')
    plt.ylabel('Power Spectral Density (dB)')
    plt.xlim((2, 50))

    plt.legend()
    plt.show()