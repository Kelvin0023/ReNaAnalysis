import pickle

from RenaAnalysis import r_square_test
from eye.eyetracking import Fixation
from learning.models import EEGInceptionNet
from src.learning.train import train_model, epochs_to_class_samples, eval_model
from params import conditions, dtnn_types

# analysis parameters ######################################################################################
participant = '1'
session = 1

print("Loading RDF")
rdf = pickle.load(open('rdf.p', 'rb'))

print('Training model on constrained blocks')
# event_names = ["Distractor", "Target"]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
# x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True, participant=participant, session=session)
# pickle.dump(x, open('x.p', 'wb'))
# pickle.dump(y, open('y.p', 'wb'))

x = pickle.load(open('x.p', 'rb'))
y = pickle.load(open('y.p', 'rb'))
model = EEGInceptionNet(in_shape=x.shape, num_classes=2)
model, training_histories, criterion, label_encoder = train_model(x, y, model)

# load the free viewing epochs

event_names = ["Distractor", "Target"]
event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
# r_square_test(rdf, event_names, event_filters, title="Visual Search epochs locked to detected fixation (I-VT-Head)")
x_i_dt_head, y_i_dt_head, epochs, event_ids, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type='eeg', rebalance=True, participant=participant, session=session)
loss, accuracy = eval_model(model, x_i_dt_head, y_i_dt_head, criterion, label_encoder)