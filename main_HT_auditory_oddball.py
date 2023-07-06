import datetime
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from renaanalysis.learning.train import eval_model, preprocess_model_data
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples


result_path = 'results/model_performances_auditory_oddball'
# analysis parameters ######################################################################################
eeg_resample_rate = 200
reject = 'auto'

bids_root = 'D:/Dropbox/Dropbox/ReNa/EEGDatasets/auditory_oddball_openneuro'
# bids_root = 'D:/Dataset/auditory_oddball'

event_names = ["standard", "oddball_with_reponse"]
colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}
picks = 'eeg'
# models = ['HT', 'HDCA', 'EEGCNN']
models = ['HT-pca-ica']
n_folds = 6
ht_lr = 1e-3
ht_l2 = 1e-5

reload_saved_samples = False
viz_rebalance = False
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

x, y, start_time, metadata = get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, eeg_resample_rate, colors)


# lockings test  ####################################################################################################
results = dict()
if os.path.exists(f'{export_data_root}/x_pca_ica.p') and os.path.exists(f'{export_data_root}/x_znormed.p'):
    with open(f'{export_data_root}/x_pca_ica.p', "rb") as file:
        x_eeg_pca_ica = pickle.load(file)
    with open(f'{export_data_root}/x_znormed.p', "rb") as file:
        x_eeg_znormed = pickle.load(file)
    with open(f'{export_data_root}/pca.p', "rb") as file:
        pca = pickle.load(file)
    with open(f'{export_data_root}/ica.p', "rb") as file:
        ica = pickle.load(file)
    x_pupil_znormed = None
else:
    x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica  = preprocess_model_data(x, None)
    with open(f'{export_data_root}/x_pca_ica.p', "wb") as file:
        pickle.dump(x_eeg_pca_ica, file)
    with open(f'{export_data_root}/x_znormed.p', "wb") as file:
        pickle.dump(x_eeg_znormed, file)
    with open(f'{export_data_root}/pca.p', "wb") as file:
        pickle.dump(pca, file)
    with open(f'{export_data_root}/ica.p', "wb") as file:
        pickle.dump(ica, file)

now = datetime.datetime.now()
datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")
result_path = result_path + datetime_string

pickle.dump(results, open(result_path, 'wb'))

for m in models:
    m_results, training_histories = eval_model(x, None, y, event_names, model_name=m, exg_resample_rate=eeg_resample_rate, n_folds=n_folds, ht_lr=ht_lr, ht_l2=ht_l2, eeg_montage=eeg_montage,
                           x_eeg_znormed=x_eeg_znormed, x_eeg_pca_ica=x_eeg_pca_ica, x_pupil_znormed=x_pupil_znormed,
                           test_name=f"auditory_oddball_{m}_{datetime_string}", viz_rebalance=viz_rebalance, pca=pca, ica=ica)
    results = {**m_results, **results}
    pickle.dump(results, open(result_path, 'wb'))
    pickle.dump(training_histories, open(result_path + f'{m}_training_history', 'wb'))


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
# plt.rcParams["figure.figsize"] = (24, 12)

models = ['HDCA_EEG', 'HT', 'EEGCNN']

width = 0.175
metric = 'test auc'

metric_values = [results[m][metric] for m_index, m in enumerate(models)]  # get the auc for each locking
plt.bar(np.arange(len(metric_values)), metric_values, width, label=f'{metric}')
for j in range(len(metric_values)):
    plt.text(j +  width, metric_values[j] + 0.05, str(round(metric_values[j], 3)), horizontalalignment='center',verticalalignment='center')

plt.ylim(0.0, 1.1)
plt.ylabel(f'{metric} (averaged across folds)')
plt.title(f'Auditory oddball {metric}')
plt.xticks(np.arange(len(metric_values)), models)
plt.legend()
plt.show()