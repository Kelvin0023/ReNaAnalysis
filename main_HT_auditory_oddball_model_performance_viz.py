import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from renaanalysis.learning.result_viz import viz_model_performance
from renaanalysis.learning.train import eval_model
from renaanalysis.learning.preprocess import preprocess_samples_eeg_pupil
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples

# analysis parameters ######################################################################################


models = ['HDCA_EEG', 'EEGCNN', 'HT', 'HT-pca-ica']
results = pickle.load(open('results/model_performances_auditory_oddball_5_16_2023_all', 'rb'))

metrics = ['test auc', 'folds val auc', 'folds val acc', 'folds train acc']

viz_model_performance(results, metrics, models, width=0.125)