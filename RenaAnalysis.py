import os
import time

import torch
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import math

from eye.eyetracking import gaze_event_detection_I_VT, gaze_event_detection_PatchSim
from learning.train import epochs_to_class_samples, prepare_sample_label
from params import *
from utils.RenaDataFrame import RenaDataFrame
from utils.fs_utils import load_participant_session_dict, get_data_file_paths, get_analysis_result_paths
from utils.utils import get_item_events, visualize_pupil_epochs


# def eeg_event_discriminant_analysis(rdf: RenaDataFrame, event_names, event_filters, participant=None, session=None):
#     eeg_epochs, eeg_event_ids = rdf.get_eeg_epochs(event_names, event_filters, participant, session)

def r_square_test(rdf: RenaDataFrame, event_names, event_filters, tmin_eeg=-0.1, tmax_eeg=1.0, participant=None, session=None, title="", fig_size=(25.6, 14.4)):
    plt.rcParams["figure.figsize"] = fig_size
    colors = {'Distractor': 'blue', 'Target': 'red'}
    plt.rcParams.update({'font.size': 22})
    assert len(event_names) == len(event_filters) == 2
    x, y, _ = prepare_sample_label(rdf, event_names, event_filters, eeg_picks, tmin_eeg=tmin_eeg, tmax_eeg=tmax_eeg)
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
                events += gaze_event_detection_I_VT(data['Unity.VarjoEyeTrackingComplete'], events,
                                                    headtracking_data_timestamps=data['Unity.HeadTracker'])
                # add gaze behaviors from patch sim
                events += gaze_event_detection_PatchSim(data['FixationDetection'][0], data['FixationDetection'][1],
                                                        events)

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

def compute_forward(x_windowed, y, projection):
    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_windowed.shape
    activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
    for class_index in range(2):  # for test set
        this_x = x_windowed[y == class_index]
        this_projection = projection[y == class_index]
        for j in range(num_windows):
            this_x_window = this_x[:, :, j, :].reshape(this_x.shape[0], -1).T
            # z_window = np.array([np.dot(weights_channelWindow[j], this_x[trial_index, :, j, :].reshape(-1)) for trial_index in range(this_x.shape[0])])
            # z_window = z_window.reshape((-1, 1)) # change to a col vector
            this_projection_window = this_projection[:, j]
            a = (np.matmul(this_x_window, this_projection_window) / np.matmul(this_projection_window.T, this_projection_window).item()).reshape((num_channels, num_timepoints_per_window))
            activation[class_index, :, j] = a
    return activation

def plot_forward(activation, event_names, split_window, num_windows, notes):
    info = mne.create_info(
        eeg_channel_names,
        sfreq=exg_resample_srate,
        ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)

    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    # fig, axs = plt.subplots(2, num_windows - 1, figsize=(22, 10), sharey=True)  # sharing vmax and vmin
    for class_index, e_name in enumerate(event_names):
        axes = subfigs[class_index].subplots(1, num_windows, sharey=True)
        for i in range(num_windows):
            a = np.mean(activation[class_index, :, i, :], axis=1)
            plot_topomap(a, info, axes=axes[i - 1], show=False, res=512, vlim=(np.min(activation), np.max(activation)))
            axes[i - 1].set_title(f"{int((i - 1) * split_window * 1e3)}-{int(i * split_window * 1e3)}ms")
        subfigs[class_index].suptitle(e_name)
    fig.suptitle(f"Activation map from Fisher Discriminant Analysis. {notes}", fontsize='x-large')
    plt.show()

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out

def solve_crossbin_weights(projection_train, projection_test, y_train, y_test, num_windows, method='torch'):
    if method == 'torch':
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_path = os.path.join(model_save_dir, 'best_logistic_regression_cross_bin')
        _y_train = torch.Tensor(np.expand_dims(y_train, axis=1)).to(device)
        _y_test = torch.Tensor(np.expand_dims(y_test, axis=1)).to(device)
        # if remove_first:
        #     # get rid of the first bin, which is before target onset
        #     projection_train = projection_train[:, 1:]
        #     projection_test = projection_test[:, 1:]
        #     num_windows = num_windows-1
        _projectionTrain_window_trial = torch.Tensor(projection_train).to(device)
        _projectionTest_window_trial = torch.Tensor(projection_test).to(device)
        model = linearRegression(num_windows, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()
        best_roc_auc = -np.inf

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = torch.sigmoid(model(_projectionTrain_window_trial))
            l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()])
            loss = criterion(y_pred, _y_train) + l2_penalty
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = torch.sigmoid(model(_projectionTest_window_trial))
                l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()])
                loss_test = criterion(y_pred, _y_test) + l2_penalty
                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred.detach().cpu().numpy())
                test_roc_auc = metrics.auc(fpr, tpr)
            # print(f"epoch {epoch}, train loss is {loss.item()}, test loss is {loss_test.item()}, test auc is {test_roc_auc}")

            if test_roc_auc > best_roc_auc:
                torch.save(model.state_dict(), model_path)
                # print(f'Best model auc improved from {best_roc_auc} to {test_roc_auc}, saved best model to {model_save_dir}')
                best_roc_auc = test_roc_auc
        # load the best model back
        best_model = linearRegression(num_windows, 1).to(device)
        best_model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            y_pred = torch.sigmoid(best_model(_projectionTest_window_trial))
            y_pred = y_pred.detach().cpu().numpy()
            cross_window_weights = model.linear.weight.detach().cpu().numpy()[0, :]
    else:
        model = LogisticRegression(random_state=random_seed, max_iter=epochs, fit_intercept=False, penalty='l2', solver='liblinear').fit(projection_train, y_train)
        y_pred = model.predict(projection_test)
        cross_window_weights = np.squeeze(model.coef_, axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    # fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    # display.plot(ax=plt.gca())
    # plt.tight_layout()
    # plt.show()

    # plt.plot(cross_window_weights)
    # plt.xticks(ticks=list(range(1, num_windows + 1)), labels=[str(x) for x in list(range(1, num_windows + 1))])
    # plt.xlabel("100ms windowed bins")
    # plt.ylabel("Cross-bin weights")
    # plt.tight_layout()
    # plt.show()
    return cross_window_weights, roc_auc, fpr, tpr


def compute_window_projections(x_train_windowed, x_test_windowed, y_train, num_top_components=20):
    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_train_windowed.shape
    num_test_trials = len(x_test_windowed)

    weights_channelWindow = np.empty((num_windows, num_channels * num_timepoints_per_window))
    projectionTrain_window_trial = np.empty((num_train_trials, num_windows))
    projectionTest_window_trial = np.empty((num_test_trials, num_windows))
    # TODO this can go faster with multiprocess pool
    for k in range(num_windows):  # iterate over different windows
        this_x_train = x_train_windowed[:, :, k, :].reshape((num_train_trials, -1))
        this_x_test = x_test_windowed[:, :, k, :].reshape((num_test_trials, -1))
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(this_x_train, y_train)
        _weights = np.squeeze(lda.coef_, axis=0)
        weights_channelWindow[k] = _weights
        for j in range(num_train_trials):
            projectionTrain_window_trial[j, k] = np.dot(_weights, this_x_train[j])
        for j in range(num_test_trials):
            projectionTest_window_trial[j, k] = np.dot(_weights, this_x_test[j])
    return weights_channelWindow, projectionTrain_window_trial, projectionTest_window_trial

def HDCA():
    pass