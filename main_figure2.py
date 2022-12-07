import os
import pickle
import time


from RenaAnalysis import get_rdf, r_square_test
from eye.eyetracking import Fixation
from params import *
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event
import os
import time
from collections import defaultdict

from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim, Fixation, GazeRayIntersect
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_analysis_result_paths, get_data_file_paths
from utils.utils import get_item_events, viz_pupil_epochs, viz_eeg_epochs
import matplotlib.pyplot as plt
import numpy as np
# analysis parameters ######################################################################################
from utils.viz_utils import visualize_gaze_events, visualize_rdf_gaze_event
import matplotlib.pyplot as plt

# end of setup parameters, start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

rdf = pickle.load(open('rdf.p', 'rb'))

# discriminant test  ####################################################################################################

plt.rcParams.update({'font.size': 22})
colors = {'Distractor': 'blue', 'Target': 'red', 'Novelty': 'orange'}

event_filters = [lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Distractor"],
                 lambda x: type(x)==Fixation and x.block_condition == conditions['VS'] and x.detection_alg == 'Patch-Sim' and x.dtn==dtnn_types["Target"]]
viz_pupil_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='VS ERP, locked to Detected Fixation (using Patch-Sim)')
viz_eeg_epochs(rdf, ["Distractor", "Target"], event_filters, colors, title='VS ERP, locked to Detected Fixation (using Patch-Sim)')
