# analysis parameters ######################################################################################
import copy
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.learning.result_viz import viz_performances_rena
from renaanalysis.learning.train_rena import eval_lockings
from renaanalysis.params.params import *


# user parameters
exg_resample_rate = 200


# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

# rdf = get_rdf(exg_resample_rate=exg_resample_rate, ocular_artifact_mode='proxy')
rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
# pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")

# lockings test  ####################################################################################################

plt.rcParams.update({'font.size': 22})

colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_names = ["Distractor", "Target"]
locking_name_filters_vs = {
                        'VS-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                        'VS-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]],
                        'VS-I-DT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]],
                       'VS-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                 lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]}

locking_name_filters_constrained = {
                        'RSVP-Item-Onset': [lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                            lambda x: x.block_condition == conditions['RSVP'] and x.dtn_onffset and x.dtn == dtnn_types["Target"]],

                        'Carousel-Item-Onset': [lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
                                                lambda x: x.block_condition == conditions['Carousel'] and x.dtn_onffset and x.dtn==dtnn_types["Target"]],

                        'RSVP-I-VT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-VT-Head' and x.dtn==dtnn_types["Target"]],
                        'RSVP-FLGI': [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['RSVP']  and x.dtn==dtnn_types["Target"]],
                        'RSVP-I-DT-Head': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['RSVP'] and x.detection_alg == 'I-DT-Head' and x.dtn==dtnn_types["Target"]],
                        'RSVP-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                 lambda x: type(x) == Fixation and x.is_first_long_gaze  and x.block_condition == conditions['RSVP'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]],

                        'Carousel-I-VT-Head': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Distractor"],
                                                lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-VT-Head' and x.dtn == dtnn_types["Target"]],
                        'Carousel-FLGI': [lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Distractor"],
                                        lambda x: type(x) == GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.dtn == dtnn_types["Target"]],
                        'Carousel-I-DT-Head': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-DT-Head' and x.dtn == dtnn_types["Distractor"],
                                        lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'I-DT-Head' and x.dtn == dtnn_types["Target"]],
                        'Carousel-Patch-Sim': [lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Distractor"],
                                                lambda x: type(x) == Fixation and x.is_first_long_gaze and x.block_condition == conditions['Carousel'] and x.detection_alg == 'Patch-Sim' and x.dtn == dtnn_types["Target"]]
                                    } #nyamu <3

# models = ['HT']
n_folds = 1
models = ['HDCA', 'HT', 'EEGCNN', 'EEGPupilCNN']

results = dict()

is_regenerate_epochs = True
for m in models:
    m_results = eval_lockings(rdf, event_names, locking_name_filters_constrained, model_name=m, regenerate_epochs=is_regenerate_epochs, exg_resample_rate=exg_resample_rate, n_folds=n_folds)
    is_regenerate_epochs = False  # dont regenerate epochs after the first time
    results = {**m_results, **results}
    pickle.dump(results, open('model_locking_performances', 'wb'))

is_regenerate_epochs = True
for m in models:
    m_results = eval_lockings(rdf, event_names, locking_name_filters_vs, participant='1', session=2, model_name=m, regenerate_epochs=is_regenerate_epochs, exg_resample_rate=exg_resample_rate)
    is_regenerate_epochs = False  # dont regenerate epochs after the first time
    results = {**m_results, **results}
    pickle.dump(results, open('model_locking_performances', 'wb'))



# import pickle

# results = pickle.load(open('model_locking_performances', 'rb'))
# new_results = dict()
# for key, value in results.items():
#     new_key = copy.copy(key)
#     if type(key[1]) is not str:
#         i = str(key[1]).index('(')
#         new_key = (new_key[0], str(new_key[1])[:i])
#     new_results[new_key] = value
# pickle.dump(new_results, open('model_locking_performances', 'wb'))
# exit()

models = ['HDCA_EEG-Pupil', 'HDCA_EEG', 'HDCA_Pupil', 'HT', 'EEGCNN', 'EEGPupilCNN']

constrained_conditions = ['RSVP', 'Carousel']
conditions_names = ['RSVP', 'Carousel', 'VS']
constrained_lockings = ['Item-Onset']
lockings = ['I-DT-Head', 'I-VT-Head', 'FLGI', 'Patch-Sim']

width = 0.175
viz_performances_rena('folds val auc', results, models, conditions_names, lockings, constrained_conditions, constrained_lockings, width=width)
viz_performances_rena('test auc', results, models, conditions_names, lockings, constrained_conditions, constrained_lockings, width=width)
