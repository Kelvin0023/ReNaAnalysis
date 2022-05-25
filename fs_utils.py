import os
import pickle

from rena.utils.data_utils import RNStream


def load_participant_session_dict(participant_session_dict, is_data_preloaded, is_save_loaded_data, preloaded_dats_path):
    if not is_data_preloaded:
        print("Preloading .dats")  # TODO parallelize loading of .dats
        for p_i, (participant_index, session_dict) in enumerate(participant_session_dict.items()):
            print("Working on participant-code[{0}]: {2} of {1}".format(int(participant_index),
                                                                        len(participant_session_dict), p_i + 1))
            for session_index, session_files in session_dict.items():
                print("Session {0} of {1}".format(session_index + 1, len(session_dict)))
                data_path, item_catalog_path, session_log_path, session_ICA_path = session_files
                if os.path.exists(
                        data_path.replace('dats', 'p')):  # load pickle if it's available as it is faster than dats
                    data = pickle.load(open(data_path.replace('dats', 'p'), 'rb'))
                else:
                    data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
                participant_session_dict[participant_index][session_index][0] = data
        # save the preloaded .dats
        if is_save_loaded_data:
            print("Saving preloaded sessions...")
            pickle.dump(participant_session_dict, open(preloaded_dats_path, 'wb'))
    else:
        print("Loading preloaded sessions...")
        participant_session_dict = pickle.load(open(preloaded_dats_path, 'rb'))
    return participant_session_dict

# def save_epoch_dict(epoch_dict, file_path):
