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
from renaanalysis.learning.train import eval_lockings, eval_models
from renaanalysis.params.params import *
from renaanalysis.utils.data_utils import epochs_to_class_samples
from renaanalysis.utils.dataset_utils import load_auditory_oddball_data

# user parameters
eeg_resample_rate = 200
reject = None
bids_root = 'D:/Dropbox/Dropbox/ReNa/EEGDatasets/auditory_oddball_openneuro'
event_names = ["standard", "oddball_with_reponse"]
colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}
picks = 'eeg'
models = ['HT', 'HDCA', 'EEGCNN']
n_folds = 10

reload_saved_samples = True
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()  # record the start time of the analysis

if not reload_saved_samples:
    subjects = load_auditory_oddball_data(bids_root=bids_root)
    all_epochs = []
    for subject_key, run_values in subjects.items():
        for run_key, run in run_values.items():
            all_epochs.append(run)
    all_epochs = mne.concatenate_epochs(all_epochs)
    x, y = epochs_to_class_samples(all_epochs, event_names, picks=picks, reject=reject, n_jobs=16, eeg_resample_rate=eeg_resample_rate, colors=colors)

    pickle.dump(x, open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'wb'))
    pickle.dump(y, open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'wb'))
else:
    x = pickle.load(open(os.path.join(export_data_root, 'x_auditory_oddball.p'), 'rb'))
    y = pickle.load(open(os.path.join(export_data_root, 'y_auditory_oddball.p'), 'rb'))

print(f"Load data took {time.time() - start_time} seconds")

# lockings test  ####################################################################################################

results = dict()

is_regenerate_epochs = True
for m in models:
    m_results = eval_models(x, y, event_names, model_name=m, exg_resample_rate=eeg_resample_rate, n_folds=n_folds)
    results = {**m_results, **results}

# is_regenerate_epochs = True
# for m in models:
#     m_results = eval_lockings(rdf, event_names, locking_name_filters_vs, participant='1', session=2, model_name=m, regenerate_epochs=is_regenerate_epochs, exg_resample_rate=exg_resample_rate)
#     is_regenerate_epochs = False  # dont regenerate epochs after the first time
#     results = {**m_results, **results}
#
# pickle.dump(results, open('model_locking_performances', 'wb'))


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

models = ['HDCA EEG', 'HDCA Pupil', 'HDCA EEG-Pupil', 'EEGPupilCNN', 'EEGCNN']
constrained_conditions = ['RSVP', 'Carousel']
conditions_names = ['RSVP', 'Carousel', 'VS']
constrained_lockings = ['Item-Onset', 'I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']
lockings = ['I-VT', 'I-VT-Head', 'FLGI', 'Patch-Sim']

width = 0.175

for c in conditions_names:
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