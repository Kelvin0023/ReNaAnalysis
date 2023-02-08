import os
import pickle
import time

import torch

from RenaAnalysis import get_rdf, r_square_test
from renaanalysis.eye.eyetracking import Fixation
from renaanalysis.learning.train import eval_model, train_model
from renaanalysis.utils.data_utils import epochs_to_class_samples
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
import os
import time

from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, Fixation, GazeRayIntersect
from renaanalysis.params.params import *
from renaanalysis.utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from renaanalysis.utils.utils import viz_pupil_epochs, viz_eeg_epochs
import matplotlib.pyplot as plt
import numpy as np
from renaanalysis.utils.viz_utils import visualize_gaze_events, visualize_block_gaze_event
import matplotlib.pyplot as plt



# analysis parameters ######################################################################################

'''
modify the selected_locking variable to select from one of the lockings below
If using locking with the prefix VS (meaning it's from the visual search condition), it is recommended to choose participant 1, session 2
'''

selected_locking = 'RSVP-Item-Onset'
export_data_root = '/data'
is_regenerate_rdf = False


# start of the main block ##############################################################################
torch.manual_seed(random_seed)
np.random.seed(random_seed)
start_time = time.time()  # record the start time of the analysis

if is_regenerate_rdf:
    rdf = get_rdf()
    if os.path.exists(export_data_root):
        os.mkdir(export_data_root)
    pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))  # dump to the SSD c drive
else:
    try:
        rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
    except FileNotFoundError:
        raise Exception(f"rdf file not found at {os.path.join(export_data_root, 'rdf.p')}, please set is_regenerate_rdf to True to generate it.")
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# Demo scripts  ########################################################################################

plt.rcParams.update({'font.size': 22})

colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}
event_names = ["Distractor", "Target"]


locking_filters = {
                    'VS-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                    'VS-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]],
                    'VS-I-VT': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]],
                   'VS-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                             lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]],
                    'RSVP-Item-Onset': [lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                        lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Target"]],
                    'Carousel-Item-Onset': [lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                            lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Target"]],
                    'RSVP-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                    'RSVP-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP']  and x.dtn==dtnn_types["Target"]],
                    'RSVP-I-VT': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                            lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]],
                   'RSVP-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                             lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]],

                    'Carousel-I-VT-Head': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Distractor"],
                                            lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Target"]],
                    'Carousel-FLGI': [lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Distractor"],
                                    lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Target"]],
                    'Carousel-I-VT': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT' and x.dtn == dtnn_types["Distractor"],
                                    lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT' and x.dtn == dtnn_types["Target"]],
                    'Carousel-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                            lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]} #nyamu <3

event_filters = locking_filters[selected_locking]

viz_eeg_epochs(rdf, event_names, event_filters, colors, title=f'{selected_locking}')
viz_pupil_epochs(rdf, event_names, event_filters, colors, title=f'{selected_locking}')
r_square_test(rdf, event_names, event_filters, title=f'{selected_locking}')

# x, y, _, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True, participant='1', session=2)

# pickle.dump(x, open(f'x_p1_s2_{event_locking}.p', 'wb'))
# pickle.dump(y, open(f'y_p1_s2_{event_locking}.p', 'wb'))

# x = pickle.load(open(f'x_p1_s2_{event_locking}.p', 'rb'))
# y = pickle.load(open(f'y_p1_s2_{event_locking}.p', 'rb'))

# model = EEGCNN(in_shape=x.shape, num_classes=2)
# model, training_histories, criterion, label_encoder = train_model(x, y, model, test_name=f'Locked to {event_locking}, P1, S2, Visaul Search')

# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to first long gaze ray intersect")

