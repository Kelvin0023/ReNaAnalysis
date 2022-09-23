import os
import pickle
import time
from collections import defaultdict
import numpy as np
from rena.utils.data_utils import RNStream

#################################################################################################
from utils.utils import flatten_list

data_root = "C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/Subjects"

# end of setup parameters, start of the main block ######################################################
start_time = time.time()
participant_list = os.listdir(data_root)
participant_directory_list = [os.path.join(data_root, x) for x in participant_list]

participant_session_dict = defaultdict(dict)  # create a dict that holds participant -> sessions -> list of sessionFiles
participant_condition_epoch_dict = defaultdict(dict)  # create a dict that holds participant -> condition epochs
for participant, participant_directory in zip(participant_list, participant_directory_list):
    file_names = os.listdir(participant_directory)
    # assert len(file_names) % 3 == 0
    # must have #files divisible by 3. That is, we have a itemCatalog, SessionLog and data file for each experiment session.
    num_sessions = flatten_list([[int(s) for s in txt if s.isdigit()] for txt in file_names])
    num_sessions = len(np.unique(num_sessions))
    for i in range(num_sessions):
        participant_session_dict[participant][i] = [os.path.join(participant_directory, x) for
                                                    x in ['{0}.dats'.format(i),
                                                          '{0}_ReNaItemCatalog.json'.format(
                                                              i),
                                                          '{0}_ReNaSessionLog.json'.format(
                                                              i)]]

# preload all the .dats

print("Preloading .dats")  # TODO parallelize loading of .dats
for participant_index, session_dict in participant_session_dict.items():
    print("Working on participant {0} of {1}".format(int(participant_index) + 1, len(participant_session_dict)))
    for session_index, session_files in session_dict.items():
        print("Session {0} of {1}".format(session_index + 1, len(session_dict)))
        data_path, item_catalog_path, session_log_path = session_files
        data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
        # now save the data back as pickle
        pickle.dump(data, open(data_path.replace('dats', 'p'), 'wb'))
dats_loading_end_time = time.time()
print("Loading data took {0} seconds".format(dats_loading_end_time - start_time))
