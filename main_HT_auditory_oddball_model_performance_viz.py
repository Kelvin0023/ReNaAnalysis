import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from renaanalysis.learning.result_viz import viz_model_performance
from renaanalysis.learning.train import eval_model, preprocess_model_data
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples

# analysis parameters ######################################################################################


models = ['HDCA_EEG', 'HT', 'EEGCNN']
results = pickle.load(open('model_locking_performances_auditory_oddball', 'rb'))
metrics = ['test auc', 'folds val auc', 'folds val acc', 'folds train acc']

viz_model_performance(results, metrics, models, width=0.125)