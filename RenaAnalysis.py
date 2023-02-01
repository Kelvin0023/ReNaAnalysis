import math
# analysis parameters ######################################################################################
import os
import time

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import torch
from mne.viz import plot_topomap
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from renaanalysis.eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from renaanalysis.learning.train import prepare_sample_label
from renaanalysis.params.params import *
from renaanalysis.utils.RenaDataFrame import RenaDataFrame
from renaanalysis.utils.data_utils import compute_pca_ica, z_norm_projection, rebalance_classes, epochs_to_class_samples
from renaanalysis.utils.fs_utils import load_participant_session_dict, get_data_file_paths, get_analysis_result_paths
from renaanalysis.utils.utils import get_item_events, visualize_pupil_epochs


# def eeg_event_discriminant_analysis(rdf: RenaDataFrame, event_names, event_filters, participant=None, session=None):
#     eeg_epochs, eeg_event_ids = rdf.get_eeg_epochs(event_names, event_filters, participant, session)

def r_square_test(rdf: RenaDataFrame, event_names, event_filters, tmin_eeg=-0.1, tmax_eeg=1.0, participant=None, session=None, title="", fig_size=(25.6, 14.4)):
    plt.rcParams["figure.figsize"] = fig_size
    colors = {'Distractor': 'blue', 'Target': 'red'}
    plt.rcParams.update({'font.size': 22})
    assert len(event_names) == len(event_filters) == 2
    x, y = prepare_sample_label(rdf, event_names, event_filters, picks=eeg_picks, data_type='eeg', tmin_eeg=tmin_eeg, tmax_eeg=tmax_eeg)
    r_square_grid = np.zeros(x.shape[1:])
    d_prime_grid = np.zeros(x.shape[1:])

    for channel_i in range(r_square_grid.shape[0]):
        for time_i in range(r_square_grid.shape[1]):
            x_train = x[:, channel_i, time_i].reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_train, y)
            d_prime_grid[channel_i, time_i] = compute_d_prime(y_true=y, y_pred=model.predict(x_train))
            r_square_grid[channel_i, time_i] = model.score(x_train, y)

    xtick_labels = [f'{int(x)} ms' for x in eeg_epoch_ticks * 1e3]
    xticks_locations = (eeg_epoch_ticks - tmin_eeg) * exg_resample_srate
    plt.xticks(xticks_locations, xtick_labels)
    plt.yticks(list(range(r_square_grid.shape[0])), eeg_picks)
    plt.imshow(r_square_grid, aspect='auto', cmap='Blues')
    plt.title("EEG coefficient of determination (r²) between target and distractor, " + title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    xtick_labels = [f'{int(x)} ms' for x in eeg_epoch_ticks * 1e3]
    xticks_locations = (eeg_epoch_ticks - tmin_eeg) * exg_resample_srate
    plt.xticks(xticks_locations, xtick_labels)
    plt.yticks(list(range(r_square_grid.shape[0])), eeg_picks)
    plt.imshow(d_prime_grid, aspect='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
    plt.title("EEG discriminability index (d`) between target and distractor, " + title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # do the same r test for all channels
    # x, y, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, tmin_eeg=tmin, tmax_eeg=tmax)
    # r_square_grid = np.zeros(x.shape[1:])
    #
    # for channel_i in range(r_square_grid.shape[0]):
    #     for time_i in range(r_square_grid.shape[1]):
    #         x_train = x[:, channel_i, time_i].reshape(-1, 1)
    #         model = LinearRegression()
    #         model.fit(x_train, y)
    #         r_square_grid[channel_i, time_i] = model.score(x_train, y)
    #
    # pos = mne.create_info(
    #     eeg_channel_names,
    #     sfreq=exg_resample_srate,
    #     ch_types=['eeg'] * len(eeg_channel_names))
    # pos.set_montage(eeg_montage)
    # step = 0.1
    # times = np.arange(0., tmax, step)
    # for i, start_time in enumerate(times):
    #     values_to_plot = r_square_grid[:, int(start_time * exg_resample_srate) : int((start_time + step) * exg_resample_srate)]
    #     values_to_plot = np.max(values_to_plot, axis=1)
    #     plt.subplot(1, len(times), i+1)
    #     mne.viz.plot_topomap(values_to_plot, pos=pos, vmin=0, vmax=np.max(r_square_grid), show=False)
    #     plt.title(f'{start_time}-{start_time+step}')
    #     plt.show()

    # pupilometries
    x, y, epochs, event_ids = epochs_to_class_samples(rdf, event_names, event_filters, data_type='pupil', participant=participant, session=session)
    x = np.mean(x, axis=1)
    r_square_grid = np.zeros(x.shape[1])
    for time_i in range(len(r_square_grid)):
        x_train = x[:, time_i].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_train, y)
        r_square_grid[time_i] = model.score(x_train, y)

    visualize_pupil_epochs(epochs, event_ids, colors, show=False, fig_size=fig_size)
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
        participant_session_data_dict = load_participant_session_dict(participant_session_file_path_dict, preloaded_dats_path)  # load the dats files
        print("Loading data took {0} seconds".format(time.time() - start_time))

        for p_i, (participant_index, session_dict) in enumerate(participant_session_data_dict.items()):
            for session_index, session_files in session_dict.items():
                print("Processing participant-code[{0}]: {4} of {1}, session {2} of {3}".format(int(participant_index),
                                                                                                len(participant_session_data_dict),
                                                                                                session_index + 1,
                                                                                                len(session_dict),
                                                                                                p_i + 1))
                data, item_catalog_path, session_log_path, session_bad_eeg_channels_path, session_ICA_path, video_path = session_files
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
                events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events, headtracking_data_timestamps=data['Unity.HeadTracker'])
                # add gaze behaviors from patch sim
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1], events)

                # visualize_gaze_events(events, 6)
                rdf.add_participant_session(data, events, participant_index, session_index, session_bad_eeg_channels,
                                            session_ICA_path, video_path)  # also preprocess the EEG data

    rdf.preprocess()
    end_time = time.time()
    print("Getting RDF Took {0} seconds".format(end_time - start_time))
    return rdf

def compute_d_prime(y_true, y_pred):
    Z = norm.ppf
    return math.sqrt(2) * Z(roc_auc_score(y_true, y_pred))

