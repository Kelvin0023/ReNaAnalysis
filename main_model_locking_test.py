# analysis parameters ######################################################################################
import copy
import os
import pickle
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch

from RenaAnalysis import get_rdf
from renaanalysis.eye.eyetracking import Fixation, GazeRayIntersect
from renaanalysis.learning.train import eval_lockings
from renaanalysis.params.params import *

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = get_rdf()
# rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))
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
                        'VS-I-VT': [lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Distractor"],
                                lambda x: type(x)==Fixation and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.detection_alg == 'I-VT' and x.dtn==dtnn_types["Target"]],
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

models = ['HDCA', 'EEGPupilCNN', 'EEGCNN']

results = dict()
is_regenerate_epochs = True

for m in models:
    m_results = eval_lockings(rdf, event_names, locking_name_filters_constrained, model_name=m, regenerate_epochs=is_regenerate_epochs, reduce_dim=True)
    is_regenerate_epochs = False
    results = {**m_results, **results}

is_regenerate_epochs = True
for m in models:
    m_results = eval_lockings(rdf, event_names, locking_name_filters_vs, participant='1', session=2, model_name=m, regenerate_epochs=is_regenerate_epochs, reduce_dim=True)
    is_regenerate_epochs = False
    results = {**m_results, **results}

pickle.dump(results, open('model_locking_performances', 'wb'))


# import pickle
#
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
plt.rcParams["figure.figsize"] = (24, 12)

models = ['EEGPupilCNN', 'EEGCNN', 'HDCA EEG', 'HDCA Pupil', 'HDCA EEG-Pupil']
constrained_conditions = ['RSVP', 'Carousel']
conditions = ['RSVP', 'Carousel', 'VS']
constrained_lockings = ['Item-Onset', 'I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']
lockings = ['I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']

width = 0.175

for c in conditions:
    this_lockings = lockings if c not in constrained_conditions else constrained_lockings
    ind = np.arange(len(this_lockings))

    for m_index, m in enumerate(models):
        aucs = [results[(f'{c}-{l}', m)]['folds val auc'] for l in this_lockings]  # get the auc for each locking

        plt.bar(ind + m_index * width, aucs, width, label=f'{m}')
        for j in range(len(aucs)):
            plt.text(ind[j] + m_index * width, aucs[j] + 0.05, str(round(aucs[j], 3)), horizontalalignment='center',verticalalignment='center')

    plt.ylim(0.0, 1.1)
    plt.ylabel('AUC')
    plt.title(f'Condition {c}, Accuracy by model and lockings')
    plt.xticks(ind + width / 2, this_lockings)
    plt.legend(loc=4)
    plt.show()