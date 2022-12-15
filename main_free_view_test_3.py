import os
import pickle
import time

import torch

from RenaAnalysis import get_rdf, r_square_test
from eye.eyetracking import Fixation
from learning.models import EEGInceptionNet, EEGCNN
from learning.train import epochs_to_class_samples, eval_model, train_model
from params import *
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_block_gaze_event
import os
import time
from collections import defaultdict

from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, Fixation, GazeRayIntersect
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from utils.utils import get_item_events, viz_pupil_epochs, viz_eeg_epochs
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_block_gaze_event
import matplotlib.pyplot as plt

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf()
rdf = pickle.load(open('C:/Data/rdf.p', 'rb'))
# rdf = pickle.dump('C:/Data/rdf.p', open('rdf.p', 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})

colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}
#
event_names = ["Distractor", "Target"]
# event_locking = 'I-VT'
event_locking = 'I-VT-Head'
# event_locking = 'Patch-Sim'
# event_locking = 'FLGI'

if event_locking == 'FLGI':
    event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                     lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
else:
    event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == event_locking and x.dtn==dtnn_types["Distractor"],
                     lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == event_locking and x.dtn==dtnn_types["Target"]]
# viz_eeg_epochs(rdf, event_names, event_filters, colors, title='Grid visual search, locked to long fixations', participant='1', session=2)
# viz_pupil_epochs(rdf, event_names, event_filters, colors, title='Grid visual search, locked to long fixations', participant='1', session=2)
# r_square_test(rdf, event_names, event_filters, title='Grid visual search, locked to long fixations', participant='1', session=2)

x, y, _, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True, participant='1', session=2)

pickle.dump(x, open('x_p1_s2.p', 'wb'))
pickle.dump(y, open('y_p1_s2.p', 'wb'))

# x = pickle.load(open('x_p1_s2.p', 'rb'))
# y = pickle.load(open('y_p1_s2.p', 'rb'))

model = EEGCNN(in_shape=x.shape, num_classes=2)
model, training_histories, criterion, label_encoder = train_model(x, y, model, test_name=f'Locked to {event_locking}, P1, S2, Visaul Search')

# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to first long gaze ray intersect")

