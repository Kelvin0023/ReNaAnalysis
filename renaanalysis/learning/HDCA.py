# analysis parameters ######################################################################################

# analysis parameters ######################################################################################
import os

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from mne.viz import plot_topomap
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from renaanalysis.params.params import eyetracking_resample_srate, \
    eeg_channel_names, \
    eeg_montage, model_save_dir, epochs, \
    l2_weight, random_seed, split_window_eeg, split_window_pupil, num_folds, num_top_compoenents
from renaanalysis.utils.data_utils import compute_pca_ica, z_norm_projection, rebalance_classes


def compute_forward(x_windowed, y, projection):
    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_windowed.shape
    activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
    for class_index in np.sort(np.unique(y)):  # for test set, in increasing order
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

def plot_forward(activation, event_names, split_window, num_windows, exg_srate, notes):
    info = mne.create_info(
        eeg_channel_names,
        sfreq=exg_srate,
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
    fig.suptitle(f"{notes} Activation map from Fisher Discriminant Analysis. ", fontsize='x-large')
    plt.show()

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out

def solve_crossbin_weights(projection_train, projection_test, y_train, y_test, num_windows, method='sklearn', verbose=0):
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
            if verbose: print(f"epoch {epoch}, train loss is {loss.item()}, test loss is {loss_test.item()}, test auc is {test_roc_auc}")

            if test_roc_auc > best_roc_auc:
                torch.save(model.state_dict(), model_path)
                if verbose: print(f'Best model auc improved from {best_roc_auc} to {test_roc_auc}, saved best model to {model_save_dir}')
                best_roc_auc = test_roc_auc
        # load the best model back
        best_model = linearRegression(num_windows, 1).to(device)
        best_model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            y_pred = torch.sigmoid(best_model(_projectionTest_window_trial))
            y_pred = y_pred.detach().cpu().numpy()
            cross_window_weights = model.linear.weight.detach().cpu().numpy()[0, :]
    else:
        model = LogisticRegression(random_state=random_seed, max_iter=epochs, fit_intercept=True, penalty='l2', solver='saga').fit(projection_train, y_train)
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


def compute_window_projections(x_train_windowed, x_test_windowed, y_train):
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
        projectionTrain_window_trial[:, k] = lda.fit_transform(this_x_train, y_train).squeeze(axis=1)
        projectionTest_window_trial[:, k] = lda.transform(this_x_test).squeeze(axis=1)

        _weights = np.squeeze(lda.coef_, axis=0)
        weights_channelWindow[k] = _weights
        # for j in range(num_train_trials):
        #     projectionTrain_window_trial[j, k] = np.dot(_weights, this_x_train[j]) + lda.intercept_
        # for j in range(num_test_trials):
        #     projectionTest_window_trial[j, k] = np.dot(_weights, this_x_test[j])
    return weights_channelWindow, projectionTrain_window_trial, projectionTest_window_trial


def hdca(x, y, event_names, x_type=None, is_plots=False, notes="", exg_srate=128, verbose=0):
    """

    :param x: array or list of arrays, if array, must provide if this array is EEG or pupil, if list, it is assumed that
    the array with more channels is EEG. If the two arraies have the same number of channels, an error will be raised
    :param y:
    :param event_names:
    :param is_plots:
    :param notes:
    :param verbose:
    :return:
    """

    split_size_eeg = int(split_window_eeg * exg_srate)  # split data into 100ms bins
    split_size_pupil = int(split_window_pupil * eyetracking_resample_srate)
    # multi-fold cross-validation
    cross_val_folds = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
    _, num_eeg_channels, num_windows_eeg, num_timepoints_per_window_eeg = sliding_window_view(x[0],
                                                                                              window_shape=split_size_eeg,
                                                                                              axis=2)[:, :,
                                                                          0::split_size_eeg, :].shape
    _, num_pupil_channels, num_windows_pupil, num_timepoints_per_window_pupil = sliding_window_view(x[1],
                                                                                                    window_shape=split_size_pupil,
                                                                                                    axis=2)[:, :,
                                                                                0::split_size_pupil, :].shape

    cw_weights_eeg_folds = np.empty((num_folds, num_windows_eeg))
    cw_weights_pupil_folds = np.empty((num_folds, num_windows_pupil))

    activations_folds = np.empty(
        (num_folds, len(event_names), num_eeg_channels, num_windows_eeg, num_timepoints_per_window_eeg))
    roc_auc_folds_eeg = np.empty(num_folds)
    roc_auc_folds_pupil = np.empty(num_folds)
    fpr_folds_eeg = []
    tpr_folds_eeg = []
    fpr_folds_pupil = []
    tpr_folds_pupil = []

    x_eeg_transformed, pca, ica = compute_pca_ica(x[0], num_top_compoenents)

    for i, (train, test) in enumerate(cross_val_folds.split(x[0],
                                                            y)):  # cross-validation; group arguement is not necessary unless using grouped folds
        if verbose: print(f"Working on {i + 1} fold of {num_folds}")

        x_eeg_transformed_train, x_eeg_transformed_test, y_train, y_test = x_eeg_transformed[train], x_eeg_transformed[
            test], y[train], y[test]
        x_pupil_train, x_pupil_test, _, _ = x[1][train], x[1][test], y[train], y[test]

        x_eeg_transformed_train, y_train_eeg = rebalance_classes(x_eeg_transformed_train, y_train)  # rebalance by class
        x_pupil_train, y_train_pupil = rebalance_classes(x_pupil_train, y_train)  # rebalance by class
        assert np.all(y_train_eeg == y_train_pupil)
        y_train = y_train_pupil

        x_eeg_test = x[0][test]

        x_eeg_transformed_train_windowed = sliding_window_view(x_eeg_transformed_train, window_shape=split_size_eeg, axis=2)[:, :, 0::split_size_eeg, :]  # shape = #trials, #channels, #windows, #time points per window
        x_eeg_transformed_test_windowed = sliding_window_view(x_eeg_transformed_test, window_shape=split_size_eeg, axis=2)[:, :, 0::split_size_eeg, :]  # shape = #trials, #channels, #windows, #time points per window
        x_eeg_test_windowed = sliding_window_view(x_eeg_test, window_shape=split_size_eeg, axis=2)[:, :, 0::split_size_eeg, :]  # shape = #trials, #channels, #windows, #time points per window

        num_train_trials, num_channels_eeg, num_windows_eeg, num_timepoints_per_window_eeg = x_eeg_transformed_train_windowed.shape
        num_test_trials = len(x_eeg_transformed_test)
        # compute Fisher's LD for each temporal window
        if verbose >= 2: print("Computing windowed LDA per channel, and project per window and trial")
        weights_channelWindow_eeg, projection_train_window_trial_eeg, projection_test_window_trial_eeg = compute_window_projections(
            x_eeg_transformed_train_windowed, x_eeg_transformed_test_windowed, y_train)
        if verbose >= 2: print('Computing forward model from window projections for test set')
        activation = compute_forward(x_eeg_test_windowed, y_test, projection_test_window_trial_eeg)
        # train classifier, use gradient descent to find the cross-window weights

        # z norm
        pupil_mean = np.mean(np.concatenate([x_pupil_train, x_pupil_test], axis=0), axis=(0, 2), keepdims=True)
        pupil_std = np.std(np.concatenate([x_pupil_train, x_pupil_test], axis=0), axis=(0, 2), keepdims=True)

        x_pupil_train = (x_pupil_train - pupil_mean) / pupil_std
        x_pupil_test = (x_pupil_test - pupil_mean) / pupil_std

        x_pupil_train_windowed = sliding_window_view(x_pupil_train, window_shape=split_size_pupil, axis=2)[:, :,0::split_size_pupil,:]  # shape = #trials, #channels, #windows, #time points per window
        x_pupil_test_windowed = sliding_window_view(x_pupil_test, window_shape=split_size_pupil, axis=2)[:, :,0::split_size_pupil,:]  # shape = #trials, #channels, #windows, #time points per window
        _, num_channels_pupil, num_windows_pupil, num_timepoints_per_window_pupil = x_pupil_train_windowed.shape
        weights_channelWindow_pupil, projection_train_window_trial_pupil, projection_test_window_trial_pupil = compute_window_projections(
            x_pupil_train_windowed, x_pupil_test_windowed, y_train)
        projection_train_window_trial_pupil, projection_test_window_trial_pupil = z_norm_projection(
            projection_train_window_trial_pupil, projection_test_window_trial_pupil)

        # z-norm the projections
        projection_train_window_trial_eeg, projection_test_window_trial_eeg = z_norm_projection(projection_train_window_trial_eeg, projection_test_window_trial_eeg)

        if verbose >= 2: print('Solving cross bin weights')
        cw_weights_eeg, roc_auc_eeg, fpr_eeg, tpr_eeg = solve_crossbin_weights(projection_train_window_trial_eeg,
                                                                               projection_test_window_trial_eeg,
                                                                               y_train, y_test, num_windows_eeg)

        cw_weights_pupil, roc_auc_pupil, fpr_pupil, tpr_pupil = solve_crossbin_weights(
            projection_train_window_trial_pupil, projection_test_window_trial_pupil, y_train, y_test,
            num_windows_pupil)

        projection_combined_train = np.concatenate(
            [projection_train_window_trial_eeg, projection_train_window_trial_pupil], axis=1)
        projection_combined_test = np.concatenate(
            [projection_test_window_trial_eeg, projection_test_window_trial_pupil], axis=1)

        cw_weights_combined, roc_auc_combined, fpr_combined, tpr_combined = solve_crossbin_weights(
            projection_combined_train, projection_combined_test, y_train, y_test,
            num_windows_pupil)

        cw_weights_pupil_folds[i] = cw_weights_pupil
        roc_auc_folds_pupil[i] = roc_auc_pupil
        fpr_folds_pupil.append(fpr_pupil)
        tpr_folds_pupil.append(tpr_pupil)

        cw_weights_eeg_folds[i] = cw_weights_eeg

        activations_folds[i] = activation

        roc_auc_folds_eeg[i] = roc_auc_eeg
        fpr_folds_eeg.append(fpr_eeg)
        tpr_folds_eeg.append(tpr_eeg)
        # print(f'Fold {i}, auc is {roc_auc_folds[i]}')

    plot_forward(np.mean(activations_folds, axis=0), event_names, split_window_eeg, num_windows_eeg, exg_srate=exg_srate,
                 notes=f"{notes} Average over {num_folds}-fold's test set")

    if verbose:
        print(f"Mean EEG cross ROC-AUC is {np.mean(roc_auc_folds_eeg)}")
        print(f"Mean Pupil cross ROC-AUC is {np.mean(roc_auc_folds_pupil)}")

    if is_plots:
        best_fold_i = np.argmax(roc_auc_folds_eeg)
        display = metrics.RocCurveDisplay(fpr=fpr_folds_eeg[best_fold_i], tpr=tpr_folds_eeg[best_fold_i],
                                          roc_auc=roc_auc_folds_eeg[best_fold_i], estimator_name='example estimator')
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        display.plot(ax=plt.gca(), name='ROC')
        plt.tight_layout()
        plt.title(f"{notes} Pupil ROC of the best cross-val fold")
        plt.show()

        best_fold_i = np.argmax(roc_auc_folds_pupil)
        display = metrics.RocCurveDisplay(fpr=fpr_folds_pupil[best_fold_i], tpr=tpr_folds_pupil[best_fold_i],
                                          roc_auc=roc_auc_folds_pupil[best_fold_i], estimator_name='example estimator')
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        display.plot(ax=plt.gca(), name='ROC')
        plt.tight_layout()
        plt.title(f"{notes} EEG ROC of the best cross-val fold")
        plt.show()

        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        plt.boxplot(cw_weights_eeg_folds)
        # plt.plot(cross_window_weights)
        x_labels = [f"{int((i - 1) * split_window_eeg * 1e3)}ms" for i in range(num_windows_eeg)]
        x_ticks = np.arange(0.5, num_windows_eeg + 0.5, 1)
        plt.plot(list(range(1, num_windows_eeg + 1)), np.mean(cw_weights_eeg_folds, axis=0), label="folds average")

        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel("100 ms windowed bins")
        plt.ylabel("Cross-bin weights")
        plt.title(f'{notes} Cross-bin weights, {num_folds}-fold cross validation')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        plt.boxplot(cw_weights_pupil_folds)
        # plt.plot(cross_window_weights)
        x_labels = [f"{int((i - 1) * split_window_pupil * 1e3)}ms" for i in range(num_windows_pupil)]
        x_ticks = np.arange(0.5, num_windows_pupil + 0.5, 1)
        plt.plot(list(range(1, num_windows_pupil + 1)), np.mean(cw_weights_pupil_folds, axis=0), label="folds average")

        plt.xticks(ticks=x_ticks, labels=x_labels)
        plt.xlabel("500 ms windowed bins")
        plt.ylabel("Cross-bin weights")
        plt.title(f'{notes} Cross-bin weights, {num_folds}-fold cross validation')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return roc_auc_combined, roc_auc_eeg, roc_auc_pupil