import itertools
import json

import mne
import numpy as np

from utils.Bidict import Bidict

# base_root = "C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/"
# base_root = "/Users/Leo/Dropbox/ReNa/data/ReNaPilot-2022Fall"
base_root = "D:/Dropbox/Dropbox/ReNa/data/ReNaPilot-2022Fall"
# data_directory = "Subjects"
data_directory = "Subjects"
# data_directory = "Subjects-Test-IncompleteBlock"

# event_ids_dict = {'EventMarker': {'DistractorPops': 1, 'TargetPops': 2, 'NoveltyPops': 3},
#             'GazeRayIntersect': {'GazeRayIntersectsDistractor': 4, 'GazeRayIntersectsTarget': 5, 'GazeRayIntersectsNovelty': 6},
#             'GazeBehavior': {'FixationDistractor': 7, 'FixationTarget': 8, 'FixationNovelty': 9, 'FixationNull': 10,
#                               'Saccade2Distractor': 11, 'Saccade2Target': 12, 'Saccade2Novelty': 13,
#                               'Saccade2Null': 14},
#                   }  # event_ids_for_interested_epochs
# color_dict = {
#               'DistractorPops': 'blue', 'TargetPops': 'red', 'NoveltyPops': 'orange',
#               'Fixation': 'blue', 'Saccade': 'orange',
#                 "GazeRayIntersectsDistractor": 'blue', "GazeRayIntersectsTarget": 'red', "GazeRayIntersectsNovelty": 'orange',
#               'FixationDistractor': 'blue', 'FixationTarget': 'red', 'FixationNovelty': 'orange', 'FixationNull': 'grey',
#               'Saccade2Distractor': 'blue', 'Saccade2Target': 'red', 'Saccade2Novelty': 'orange', 'Saccade2Null': 'yellow'}

dtn_color_dict = {1: 'blue', 2: 'red', 3: 'orange', 4: 'grey'}

event_viz = 'GazeRayIntersect'


conditions = Bidict({'RSVP': 1., 'Carousel': 2., 'VS': 3., 'TS': 4., 'TS-gnd': 8, 'TS-id': 9})
dtnn_types = Bidict({'Distractor': 1, 'Target': 2, 'Novelty': 3, 'Null': 4})
meta_blocks = Bidict({'cp': 5, 'ip': 7})

varjoEyetrackingComplete_preset_path = 'presets/VarjoEyeDataComplete.json'
eeg_preset_path = 'presets/BioSemi.json'
eventmarker_preset_path = 'presets/ReNaEventMarker.json'
headtracker_preset_path = 'presets/UnityHeadTracking.json'
# load presets
varjoEyetracking_preset = json.load(open(varjoEyetrackingComplete_preset_path))
eeg_preset = json.load(open(eeg_preset_path))
eventmarker_preset = json.load(open(eventmarker_preset_path))
headtracker_preset = json.load(open(headtracker_preset_path))

tmin_pupil = -1
tmax_pupil = 3.
tmin_pupil_viz = -0.1
tmax_pupil_viz = 3.

tmin_eeg = -1.2
tmax_eeg = 2.4

tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.

eyetracking_srate = 200
exg_srate = 2048
exg_resample_srate = 128
eeg_picks = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']

eeg_montage = mne.channels.make_standard_montage('biosemi64')
eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
ecg_ch_name='ECG00'

note = "test_v3"

FIXATION_MINIMAL_TIME = 1e-3 * 141.42135623730952

START_OF_BLOCK_ENCODING = 4
END_OF_BLOCK_ENCODING = 5

# ITEM_TYPE_ENCODING = {event_ids_dict['GazeRayIntersect']['GazeRayIntersectsDistractor']: 'distractor',
#                       event_ids_dict['GazeRayIntersect']['GazeRayIntersectsTarget']: 'target',
#                       event_ids_dict['GazeRayIntersect']['GazeRayIntersectsNovelty']: 'novelty'}


'''
The core events, each core event will have some meta information associated with it

RSVP-pop: 
'''

# classifier_prep_markers = ['{}-{}-{}-{}'.format(a, b, c ,d) for a, b, c ,d in itertools.product(['practice', 'exp'], ['RSVP', 'Carousel'], ['Distractor', 'Target', 'Novelty'], ['Pop', 'IDTFixGaze', 'FixDetectGaze'])]
# identifier_prep_markers = ['{}-VS-{}-{}'.format(a, b, c) for a, b, c in itertools.product(['practice', 'exp'], ['Distractor', 'Target', 'Novelty'], ['IDTFixGaze', 'FixDetectGaze'])]
# events = ['BlockStart', 'BlockEnd'] + classifier_prep_markers + identifier_prep_markers
#
#
# events = Bidict(dict([(e, i) for i, e in enumerate(events)]))

item_marker_names = ['itemDTNType', 'ItemIndexInBlock', 'itemID', 'foveateAngle', 'isInFrustum', 'isGazeRayIntersected', 'distFromPlayer', 'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z']

SACCADE_CODE = 1
FIXATION_CODE = 2

num_items_per_constrainted_block = 30

reject = dict(eeg=100e-6)  # DO NOT reject or we will have a mismatch between EEG and pupil

is_regenerate_ica = False
debug = True

eeg_epoch_ticks = np.array([0, 0.3, 0.6, 0.8])
pupil_epoch_ticks = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3])

lr = 5e-3
batch_size = 1024
epochs = 1000
train_ratio = 0.8
model_save_dir = 'learning/saved_models'