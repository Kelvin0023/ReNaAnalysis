import os
import pickle
import time

import torch

from RenaAnalysis import get_rdf, r_square_test
from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.params.params import *
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = get_rdf()
# rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
rdf = pickle.dump(os.path.join(export_data_root, 'rdf.p'), open('rdf.p', 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# discriminant test  ####################################################################################################
event_names = ["Distractor", "Target"]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]

# statistical difference test  #####################################################
event_names = ["Distractor", "Target"]
event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                 lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="Constrained conditions (RSVP & Carousel), locked to target onset")
#
event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (Patch-Sim)", participant='1', session=2)

event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (I-VT-Head)", participant='1', session=2)

event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to detected fixation (I-VT)", participant='1', session=2)


event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                     lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
r_square_test(rdf, event_names, event_filters, title="VisualSearch epochs locked to FLGI", participant='1', session=2)
