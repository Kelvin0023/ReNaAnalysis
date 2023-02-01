import pickle

from learning.models import EEGInceptionNet, EEGCNN
from renaanalysis.learning.train import train_model
# analysis parameters ######################################################################################
import time
from collections import defaultdict

from utils.RenaDataFrame import RenaDataFrame
# analysis parameters ######################################################################################
import matplotlib.pyplot as plt

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

# rdf = pickle.load(open('rdf.p', 'rb'))

# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}


# event_names = ["Distractor", "Target"]
# event_filters = [lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS'] and x.dtn==dtnn_types["Distractor"],
#                  lambda x: type(x)==GazeRayIntersect and x.is_first_long_gaze and x.block_condition == conditions['VS']  and x.dtn==dtnn_types["Target"]]
# x, y, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True, participant='1', session=2)
#
# pickle.dump(x, open('x.p', 'wb'))
# pickle.dump(y, open('y.p', 'wb'))

x = pickle.load(open('x.p', 'rb'))
y = pickle.load(open('y.p', 'rb'))

model = EEGCNN(in_shape=x.shape, num_classes=2)
model, training_histories, criterion, label_encoder = train_model(x, y, model)

# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to first long gaze ray intersect")

