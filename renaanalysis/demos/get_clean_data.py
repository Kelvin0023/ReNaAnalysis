import json
import pickle

import mne
import numpy as np
from matplotlib import pyplot as plt

block_margin = 2.

file_path = r'D:\Dropbox\Dropbox\ReNa\data\ReNaPilot-2022Fall\12_02_2022\12_02_2022_19_39_19-Exp_RenaPilot-Sbj_zl-Ssn_2.p'
event_marker_path = 'renaanalysis/params/ReNaEventMarker.json'
eeg_preset_path = 'renaanalysis/params/BioSemi.json'
eeg_picks = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']
data = pickle.load(open(file_path, 'rb'))

event_markers = json.load(open(event_marker_path, 'r'))
eeg_preset = json.load(open(eeg_preset_path, 'r'))

block_markers = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("BlockMarker"), :]
block_ids = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("BlockIDStartEnd"), :]
dtns = data[event_markers['StreamName']][0][event_markers['ChannelNames'].index("DTN"), :]
event_timestamps = data[event_markers['StreamName']][1]
rsvp_block_start_marker_indices = np.where(block_markers == 1)[0]

rsvp_block_ids = block_ids[rsvp_block_start_marker_indices]
rsvp_block_end_marker_indices = [i for i in range(len(block_ids)) if block_ids[i] in -rsvp_block_ids]

rsvp_block_start_times = event_timestamps[rsvp_block_start_marker_indices]
rsvp_block_end_times = event_timestamps[rsvp_block_end_marker_indices]

# create the dtn stream, m stands for marker
dtn_stream = np.empty(0)
dtn_timestamps = np.empty(0)
last_timestamp = 0
for m_start, m_end in zip(rsvp_block_start_marker_indices, rsvp_block_end_marker_indices):
    dtn_stream = np.concatenate((dtn_stream, dtns[m_start:m_end]))
    block_timestamps = event_timestamps[m_start:m_end] - event_timestamps[m_start] + last_timestamp + block_margin
    last_timestamp = block_timestamps[-1]
    dtn_timestamps = np.concatenate((dtn_timestamps, block_timestamps))

montage = mne.channels.make_standard_montage('biosemi64')
pick_indices = [montage.ch_names.index(x) for x in eeg_picks]
eeg_timestamps = data['BioSemi'][1]
eeg_start_indices = [np.argmin(abs(eeg_timestamps - s)) for s in rsvp_block_start_times]
eeg_end_indices = [np.argmin(abs(eeg_timestamps - s)) for s in rsvp_block_end_times]
eeg_data = data['BioSemi'][0][1:65, :][np.array(pick_indices), :]

dtn_eeg_stream = np.empty((eeg_data.shape[0], 0))
dtn_eeg_timestamps = np.empty(0)
last_timestamp = 0

for eeg_start, eeg_end in zip(eeg_start_indices, eeg_end_indices):
    dtn_eeg_stream = np.concatenate((dtn_eeg_stream, eeg_data[:, eeg_start:eeg_end]), axis=1)
    block_timestamps = eeg_timestamps[eeg_start:eeg_end] - eeg_timestamps[eeg_start] + last_timestamp + block_margin
    last_timestamp = block_timestamps[-1]
    dtn_eeg_timestamps = np.concatenate((dtn_eeg_timestamps, block_timestamps))

dtn_eeg_stream = mne.filter.resample(dtn_eeg_stream, down=2**4)
dtn_eeg_timestamps = dtn_eeg_timestamps[::2**4]
out_data = {'Example-BioSemi-Midline': (dtn_eeg_stream, dtn_eeg_timestamps), 'Example-EventMarker': (dtn_stream[None, :], dtn_timestamps)}
pickle.dump(out_data, open(r'D:\PycharmProjects\RenaLabApp\examples/erp-example.p', 'wb'))