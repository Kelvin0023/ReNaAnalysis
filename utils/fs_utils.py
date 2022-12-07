import os
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from rena.utils.data_utils import RNStream

def read_file_lines_as_list(path):
    with open(path, 'r') as filehandle:
        out = [line.rstrip() for line in filehandle.readlines()]
    return out
def load_participant_session_dict(participant_session_dict, preloaded_dats_path):
    print("Preloading .dats")  # TODO parallelize loading of .dats
    for p_i, (participant_index, session_dict) in enumerate(participant_session_dict.items()):
        print("Loading data for participant-code[{0}]: {2} of {1}".format(int(participant_index),
                                                                    len(participant_session_dict), p_i + 1))
        for session_index, session_files in session_dict.items():
            print("Session-code [{0}]: {0} of {1}".format(session_index, session_index + 1, len(session_dict)))
            data_path = session_files[0]
            if os.path.exists(
                    data_path.replace('dats', 'p')):  # load pickle if it's available as it is faster than dats
                data = pickle.load(open(data_path.replace('dats', 'p'), 'rb'))
            else:
                print("dats file found, a .p file will be saved after streaming in for faster loading time next time")
                data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
                # if data is a .dats, save .p after loading for faster processing next time
                pickle.dump(data, open(data_path.replace('dats', 'p'), 'wb'))

            participant_session_dict[participant_index][session_index][0] = data
    # save the preloaded .dats
    # print("Saving preloaded sessions...")
    # pickle.dump(participant_session_dict, open(preloaded_dats_path, 'wb'))
    # else:
    #     print("Loading preloaded sessions...")
    #     participant_session_dict = pickle.load(open(preloaded_dats_path, 'rb'))
    return participant_session_dict

# def save_epoch_dict(epoch_dict, file_path):
def get_analysis_result_paths(base_root, note):
    dt_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    analysis_result_dir = os.path.join(base_root, 'RenaAnalysis-{}-{}'.format(dt_string, note))
    # os.mkdir(analysis_result_dir)

    preloaded_dats_path = os.path.join(analysis_result_dir, 'participant_session_dict.p')
    preloaded_epoch_path = os.path.join(analysis_result_dir, 'participant_condition_epoch_dict.p')
    preloaded_block_path = os.path.join(analysis_result_dir, 'participant_condition_block_dict.p')

    gaze_statistics_path = preloaded_epoch_path.strip('.p') + 'gaze_statistics' + '.p'
    gaze_behavior_path = preloaded_epoch_path.strip('.p') + 'gaze_behavior' + '.p'

    epoch_data_export_root = os.path.join(analysis_result_dir, 'Epochs')

    return preloaded_dats_path, preloaded_epoch_path, preloaded_block_path, gaze_statistics_path, gaze_behavior_path, epoch_data_export_root


def get_data_file_paths(base_root, data_directory):
    participant_badchannel_dict = dict()
    participant_session_file_path_dict = defaultdict(
        dict)  # create a dict that holds participant -> sessions -> list of sessionFiles

    data_root = os.path.join(base_root, data_directory)
    participant_list = [x for x in os.listdir(data_root) if x != '.DS_Store']
    participant_directory_list = [os.path.join(data_root, x) for x in participant_list if x != '.DS_Store']

    for participant, participant_directory in zip(participant_list, participant_directory_list):
        file_names = os.listdir(participant_directory)
        #  That is, we have a itemCatalog, SessionLog and data file for each experiment session.
        num_sessions = [int(txt.strip('.p')) for txt in file_names if txt != '.DS_Store' and txt.strip('.p').isdigit()]
        num_sessions = len(np.unique(num_sessions))
        if os.path.exists(os.path.join(participant_directory, 'badchannels.txt')):  # load bad channels for this participant
            participant_badchannel_dict[participant] = read_file_lines_as_list(
                os.path.join(participant_directory, 'badchannels.txt'))
        for i in range(num_sessions):
            participant_session_file_path_dict[participant][i] = [os.path.join(participant_directory, x) for
                                                                  x in ['{0}.dats'.format(i),
                                                                        '{0}_ReNaItemCatalog.json'.format(i),
                                                                        '{0}_ReNaSessionLog.json'.format(i),
                                                                        '{0}_badchannels'.format(i),
                                                                        '{0}_ParticipantSessionICA'.format(i),
                                                                        '{0}_video'.format(i)]]

    return participant_list, participant_session_file_path_dict, participant_badchannel_dict

def convert_dats_in_folder_to_p(root):
    file_paths = [os.path.join(root, x) for x in os.listdir(root)]
    for i, f_path in enumerate(file_paths):
        print(f"Working on file {f_path}, {i+1} of {len(file_paths)}")
        data = RNStream(f_path).stream_in(jitter_removal=False)
        pickle.dump(data, open(f_path.replace('dats', 'p'), 'wb'))
