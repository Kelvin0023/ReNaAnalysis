import os
import pickle
import time


from RenaAnalysis import get_rdf, r_square_test
from eye.eyetracking import Fixation
from learning.models import EEGCNNNet
from learning.train import epochs_to_class_samples, eval_model, train_model
from params import *
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event
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
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event
import matplotlib.pyplot as plt

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = get_rdf()

# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}
event_names = ["Distractor", "Target"]

# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Novelty"]]
# viz_eeg_epochs(rdf, ["Distractor", "Target", "Novelty"], event_filters, colors)
#
event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='Locked to first long Gaze Ray Intersect')

# visualize_rdf_gaze_event(rdf, participant='0', session=0, block_id=6, only_long_gaze=True)
r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to first long gaze ray intersect")

x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True)
model = EEGCNNNet(in_shape=x, num_classes=2)
model, training_histories, criterion, label_encoder = train_model(x, y, model)
# loss, accuracy = eval_model(model, x_i_dt_head, y_i_dt_head, criterion, label_encoder)