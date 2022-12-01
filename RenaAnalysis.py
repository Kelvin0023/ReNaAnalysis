import os
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from learning.train import epochs_to_class_samples
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_data_file_paths, get_analysis_result_paths
from utils.utils import get_item_events, visualize_pupil_epochs


def eeg_event_discriminant_analysis(rdf: RenaDataFrame, event_names, event_filters, participant=None, session=None):
    eeg_epochs, eeg_event_ids = rdf.get_eeg_epochs(event_names, event_filters, participant, session)


def r_square_test(rdf: RenaDataFrame, event_names, event_filters, participant=None, session=None, title="", fig_size=(25.6, 14.4)):
    plt.rcParams["figure.figsize"] = fig_size
    colors = {'Distractor': 'blue', 'Target': 'red'}
    plt.rcParams.update({'font.size': 22})
    assert len(event_names) == len(event_filters) == 2
    tmin = -0.1
    x, y = epochs_to_class_samples(rdf, event_names, event_filters, picks=eeg_picks, tmin_eeg=tmin, tmax_eeg=0.8)
    r_square_grid = np.zeros(x.shape[1:])

    for channel_i in range(r_square_grid.shape[0]):
        for time_i in range(r_square_grid.shape[1]):
            x_train = x[:, channel_i, time_i].reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_train, y)
            r_square_grid[channel_i, time_i] = model.score(x_train, y)

    xtick_labels = [f'{int(x)} ms' for x in eeg_epoch_ticks * 1e3]
    xticks_locations = (eeg_epoch_ticks - tmin) * exg_resample_srate
    plt.xticks(xticks_locations, xtick_labels)
    plt.yticks(list(range(r_square_grid.shape[0])), eeg_picks)
    plt.imshow(r_square_grid, aspect='auto', cmap='Blues')
    plt.title("EEG Statistical difference (r²) between target and distractor, " + title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # pupilometries
    pupil_epochs, pupil_event_ids = rdf.get_pupil_epochs(event_names, event_filters, participant, session)
    x, y = epochs_to_class_samples(pupil_epochs, pupil_event_ids)
    x = np.mean(x, axis=1)
    r_square_grid = np.zeros(x.shape[1])
    for time_i in range(len(r_square_grid)):
        x_train = x[:, time_i].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_train, y)
        r_square_grid[time_i] = model.score(x_train, y)

    visualize_pupil_epochs(pupil_epochs, pupil_event_ids, colors, show=False, fig_size=fig_size)
    plt.twinx()
    plt.title("Pupillometry Statistical difference (r²) between target and distractor" + title)
    xtick_labels = [f'{x} s' for x in pupil_epoch_ticks]
    xticks_locations = (pupil_epoch_ticks - tmin_pupil) * eyetracking_srate
    plt.xticks(xticks_locations, xtick_labels)
    plt.plot(np.linspace(tmin_pupil_viz, tmax_pupil_viz, len(r_square_grid)), r_square_grid, color='grey', linewidth=4)
    plt.ylim((0, 0.1))
    plt.ylabel("r²")
    plt.tight_layout()
    plt.show()



def get_rdf(is_loading_saved_analysis = False):
    start_time = time.time()  # record the start time of the analysis
    # get the list of paths to save the analysis results
    preloaded_dats_path, preloaded_epoch_path, preloaded_block_path, gaze_statistics_path, gaze_behavior_path, epoch_data_export_root = get_analysis_result_paths(
        base_root, note)
    # get the paths to the data files
    participant_list, participant_session_file_path_dict, participant_badchannel_dict = get_data_file_paths(base_root,
                                                                                                            data_directory)

    rdf = RenaDataFrame()

    if not is_loading_saved_analysis:
        participant_session_file_path_dict = load_participant_session_dict(participant_session_file_path_dict,
                                                                           preloaded_dats_path)
        print("Loading data took {0} seconds".format(time.time() - start_time))

        for p_i, (participant_index, session_dict) in enumerate(participant_session_file_path_dict.items()):
            # print("Working on participant {0} of {1}".format(int(participant_index) + 1, len(participant_session_dict)))
            for session_index, session_files in session_dict.items():
                print("Processing participant-code[{0}]: {4} of {1}, session {2} of {3}".format(int(participant_index),
                                                                                                len(participant_session_file_path_dict),
                                                                                                session_index + 1,
                                                                                                len(session_dict),
                                                                                                p_i + 1))
                data, item_catalog_path, session_log_path, session_bad_eeg_channels_path, session_ICA_path = session_files
                session_bad_eeg_channels = open(session_bad_eeg_channels_path, 'r').read().split(' ') if os.path.exists(
                    session_bad_eeg_channels_path) else None
                item_catalog = json.load(open(item_catalog_path))
                session_log = json.load(open(session_log_path))
                item_codes = list(item_catalog.values())

                # markers
                events = get_item_events(data['Unity.ReNa.EventMarkers'][0], data['Unity.ReNa.EventMarkers'][1],
                                         data['Unity.ReNa.ItemMarkers'][0], data['Unity.ReNa.ItemMarkers'][1])

                # add gaze behaviors from I-DT
                events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events)
                events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events,
                                                    headtracking_data_timestamps=data['Unity.HeadTracker'])
                # add gaze behaviors from patch sim
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1],
                                                        events)

                # visualize_gaze_events(events, 6)
                rdf.add_participant_session(data, events, participant_index, session_index, session_bad_eeg_channels,
                                            session_ICA_path)  # also preprocess the EEG data

    rdf.preprocess()
    end_time = time.time()
    print("Getting RDF Took {0} seconds".format(end_time - start_time))
    return rdf