from renaanalysis.learning.result_viz import viz_performances_rena
import pickle


results = pickle.load(open('model_locking_performances', 'rb'))

models = ['HDCA_EEG-Pupil', 'HDCA_EEG', 'HDCA_Pupil', 'HT', 'EEGCNN', 'EEGPupilCNN']

constrained_conditions = ['RSVP', 'Carousel']
conditions_names = ['RSVP', 'Carousel', 'VS']
constrained_lockings = ['Item-Onset']
lockings = ['I-DT-Head', 'I-VT-Head', 'FLGI', 'Patch-Sim']

width = 0.125
viz_performances_rena('folds val auc', results, models, conditions_names, lockings, constrained_conditions, constrained_lockings, width=width)
viz_performances_rena('test auc', results, models, conditions_names, lockings, constrained_conditions, constrained_lockings, width=width)
