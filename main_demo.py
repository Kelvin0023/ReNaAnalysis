import os
import pickle
import time

import torch

from RenaAnalysis import get_rdf, r_square_test
from renaanalysis.eye.eyetracking import Fixation
from renaanalysis.learning.HDCA import hdca
from renaanalysis.learning.models import EEGCNN, EEGPupilCNN
from renaanalysis.learning.train import eval_model, train_model, train_model_pupil_eeg
from renaanalysis.utils.data_utils import epochs_to_class_samples, compute_pca_ica, mean_max_sublists, mean_min_sublists
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
is_regenerate_rdf = True
is_regenerate_epochs = True
is_reduce_eeg_dim = True
test_name = 'demo'

# start of the main block ##############################################################################
torch.manual_seed(random_seed)
np.random.seed(random_seed)
start_time = time.time()  # record the start time of the analysis

if is_regenerate_rdf:
    rdf = get_rdf(n_jobs=20)
    if not os.path.exists(export_data_root):
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

if is_regenerate_epochs:
    x, y, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type='both', rebalance=True, participant='1', session=0)
    pickle.dump(x, open(os.path.join(export_data_root, f'x_p1_s2_{selected_locking}.p'), 'wb'))
    pickle.dump(y, open(os.path.join(export_data_root, f'y_p1_s2_{selected_locking}.p'), 'wb'))
else:
    x = pickle.load(open(os.path.join(export_data_root, f'x_p1_s2_{selected_locking}.p'), 'rb'))
    y = pickle.load(open(os.path.join(export_data_root, f'y_p1_s2_{selected_locking}.p'), 'rb'))

x_eeg = np.copy(x[0])
if is_reduce_eeg_dim:
    x[0] = compute_pca_ica(x[0], num_top_compoenents)

model = EEGPupilCNN(eeg_in_shape=x[0].shape, pupil_in_shape=x[1].shape, num_classes=2, eeg_in_channels=x[0].shape[1])
model, training_histories, criterion, label_encoder = train_model_pupil_eeg(x, y, model, test_name=test_name)
folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_val']), mean_min_sublists(training_histories['loss_val'])
folds_val_auc = mean_max_sublists(training_histories['auc_val'])
print(f'{test_name}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}')

roc_auc_combined, roc_auc_eeg, roc_auc_pupil = hdca([x_eeg, x[1]], y, event_names, is_plots=True, notes=test_name + '\n', verbose=0)  # give the original eeg data
print(f'HDCA: {test_name}: folds EEG AUC {roc_auc_eeg}, folds Pupil AUC: {roc_auc_pupil}, folds EEG-pupil AUC: {roc_auc_combined}')