# analysis parameters ######################################################################################
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from RenaAnalysis import get_rdf
from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.learning.train_rena import eval_lockings
from renaanalysis.params.params import *

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = get_rdf(n_jobs=20)
# rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# lockings test  ####################################################################################################

plt.rcParams.update({'font.size': 22})

colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_names = ["Distractor", "Target"]
locking_name_filters = {'I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                        'FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]],
                        'I-VT': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]],
                       'Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                 lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]
                        }#nyamu <3

results = eval_lockings(rdf, event_names, locking_name_filters, participant='1', session=2, model_name='EEGPupil', regenerate_epochs=True, reduce_dim=True)
pickle.dump(results, open('locking_performance', 'wb'))
