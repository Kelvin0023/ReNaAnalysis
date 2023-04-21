# analysis parameters ######################################################################################
import numpy as np
from autoreject import AutoReject
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import confusion_matrix

from renaanalysis.params.params import *
from renaanalysis.params.params import eeg_montage
from renaanalysis.utils.utils import rescale_merge_exg, visualize_eeg_epochs, visualize_pupil_epochs


# analysis parameters ######################################################################################

def get_exg_data(data):
    eeg_data = data['BioSemi'][0][1:65, :]  # take only the EEG channels
    ecg_data = data['BioSemi'][0][65:67, :]  # take only the EEG channels
    exg_data = rescale_merge_exg(eeg_data, ecg_data)  # merge and rescale eeg and ecg
    return exg_data

class Fischer:
    def __init__(self):
        """
        Inspired by https://plainenglish.io/blog/fischers-linear-discriminant-analysis-in-python-from-scratch-bbe480497504
        """
        pass


def compute_pca_ica(X, n_components, pca=None, ica=None):
    """
    data will be normaly distributed after applying this dimensionality reduction
    @param X:
    @param n_components:
    @return:
    """
    print("applying pca followed by ica")
    # ev = mne.EvokedArray(np.mean(X, axis=0),
    #                      mne.create_info(64, exg_resample_srate,
    #                                      ch_types='eeg'), tmin=-0.1)
    # ev.plot(window_title="original", time_unit='s')
    if pca is None:
        pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
        pca_data = pca.fit_transform(X)
    else:
        pca_data = pca.transform(X)
    # ev = mne.EvokedArray(np.mean(pca_data, axis=0),
    #                      mne.create_info(n_components, exg_resample_srate,
    #                                      ch_types='eeg'), tmin=-0.1)
    # ev.plot( window_title="PCA", time_unit='s')
    if ica is None:
        ica = UnsupervisedSpatialFilter(FastICA(n_components, whiten='unit-variance'), average=False)
        ica_data = ica.fit_transform(pca_data)
    else:
        ica_data = ica.transform(pca_data)
    # ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),mne.create_info(n_components, exg_resample_srate,ch_types='eeg'), tmin=-0.1)
    # ev1.plot(window_title='ICA', time_unit='s')

    return ica_data, pca, ica

def mean_sublists(l):
    return np.mean([np.mean(x) for x in l])

def mean_max_sublists(l):
    return np.mean([np.max(x) for x in l])

def mean_min_sublists(l):
    return np.mean([np.min(x) for x in l])

def z_norm_projection(x_train, x_test):
    assert len(x_train.shape) == len(x_test.shape) == 2
    projection_mean = np.mean(np.concatenate((x_train, x_test), axis=0), axis=0, keepdims=True)
    projection_std = np.std(np.concatenate((x_train, x_test), axis=0), axis=0, keepdims=True)

    return (x_train - projection_mean) / projection_std, (x_test - projection_mean) / projection_std


def rebalance_classes(x, y):
    epoch_shape = x.shape[1:]
    x = np.reshape(x, newshape=(len(x), -1))
    sm = SMOTE(random_state=random_seed)
    x, y = sm.fit_resample(x, y)
    x = np.reshape(x, newshape=(len(x),) + epoch_shape)  # reshape back x after resampling
    return x, y

def reject_combined(epochs_pupil, epochs_eeg, event_ids, n_jobs=1, n_folds=10, ar=None, return_rejections=False):
    try:
        assert len(epochs_pupil) == len(epochs_eeg)
    except AssertionError:
        raise ValueError(f'reject_combined: eeg and pupil have different number of epochs, eeg {len(epochs_eeg)}, pupil {len(epochs_pupil)}')
    if ar is not None:
        eeg_epochs_clean, log = ar.transform(epochs_eeg, return_log=True)
    else:
        ar = AutoReject(n_jobs=n_jobs, verbose=False, cv=n_folds)
        eeg_epochs_clean, log = ar.fit_transform(epochs_eeg, return_log=True)
    epochs_pupil_clean = epochs_pupil[np.logical_not(log.bad_epochs)]

    x_eeg, x_pupil, y = _epochs_to_samples_eeg_pupil(epochs_pupil_clean, eeg_epochs_clean, event_ids)
    if return_rejections:
        return x_eeg, x_pupil, y, ar, np.logical_not(log.bad_epochs)
    else:
        return x_eeg, x_pupil, y, ar


def _epochs_to_samples_eeg_pupil(epochs_pupil, epochs_eeg, event_ids, picks_eeg=None, picks_pupil=None, perserve_order=True, event_marker_to_label=True):
    y = []

    if not perserve_order:
        x_eeg = [epochs_eeg[event_name].get_data(picks=picks_eeg) for event_name, _ in event_ids.items()]
        x_pupil = [epochs_pupil[event_name].get_data(picks=picks_pupil) for event_name, _ in event_ids.items()]
        x_eeg = np.concatenate(x_eeg, axis=0)
        x_pupil = np.concatenate(x_pupil, axis=0)

        for event_name, event_class in event_ids.items():
            y += [event_class] * len(epochs_pupil[event_name].get_data())
        if event_marker_to_label:
            y = np.array(y) - 1
    else:
        y = epochs_eeg.events[:, 2]
        if event_marker_to_label: y = y - 1
        x_eeg = epochs_eeg.get_data(picks=picks_eeg)
        x_pupil = epochs_pupil.get_data(picks=picks_pupil)

    return x_eeg, x_pupil, y

def _epoch_to_samples(epochs, event_ids, picks=None, perserve_order=True, event_marker_to_label=True):
    y = []

    if not perserve_order:
        x = [epochs[event_name].get_data(picks=picks) for event_name, _ in event_ids.items()]
        x = np.concatenate(x, axis=0)

        for event_name, event_class in event_ids.items():
            y += [event_class] * len(epochs[event_name].get_data())
        if event_marker_to_label:
            y = np.array(y) - 1
    else:
        y = epochs.events[:, 2]
        if event_marker_to_label: y = y - 1
        x = epochs.get_data(picks=picks)

    return x, y

def force_square_epochs(epochs, tmin, tmax):
    # get the number of events overall, so we know what the number of time points should be to make the data matrix square
    if epochs.get_data().shape[2] != (num_epochs := epochs.get_data().shape[0]):
        target_resample_srate = num_epochs / (tmax - tmin)
        square_epochs = epochs.resample(target_resample_srate)
    return square_epochs

def epochs_to_class_samples(rdf, event_names, event_filters, *, rebalance=False, participant=None, session=None, picks=None, data_type='eeg', tmin_eeg=-0.1, tmax_eeg=0.8, eeg_resample_rate=128, tmin_pupil=-1., tmax_pupil=3., eyetracking_resample_srate=20, n_jobs=1, reject='auto', force_square=False, plots='sanity-check', colors=None, title=''):
    """
    script will always z norm along channels for the input
    @param: data_type: can be eeg, pupil or mixed
    @param: force_square: whether to call resample again on the data to force the number of epochs to match the
    number of time points. Enabling this can help algorithms that requires square matrix as their input. Default
    is disabled. Note when force_square is enabled, the resample rate (both eeg and pupil) will be ignored. rebalance
    will also be disabled.
    @param: plots: can be 'sanity_check', 'full', or none
    """
    if force_square:
        eyetracking_resample_srate = eeg_resample_rate = None
        rebalance = False
    if data_type == 'both':
        epochs_eeg, event_ids, ar_log, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session, resample_rate=eeg_resample_rate, n_jobs=n_jobs, reject=reject)
        if epochs_eeg is None:
            return None, None, None, event_ids
        epochs_pupil, event_ids, ps_group_pupil = rdf.get_pupil_epochs(event_names, event_filters, tmin=tmin_pupil, tmax=tmax_pupil, resample_rate=eyetracking_resample_srate, participant=participant, session=session, n_jobs=n_jobs)
        if reject == 'auto':  # if using auto rejection
            epochs_pupil = epochs_pupil[np.logical_not(ar_log.bad_epochs)]
            ps_group_pupil = np.array(ps_group_pupil)[np.logical_not(ar_log.bad_epochs)]
        try:
            assert np.all(ps_group_pupil == ps_group_eeg)
        except AssertionError:
            raise ValueError(f"pupil and eeg groups does not match: {ps_group_pupil}, {ps_group_eeg}")

        if force_square:
            epochs_eeg = force_square_epochs(epochs_eeg, tmin_eeg, tmax_eeg)
            epochs_pupil = force_square_epochs(epochs_pupil, tmin_pupil, tmax_pupil)
        x_eeg, x_pupil, y = _epochs_to_samples_eeg_pupil(epochs_pupil, epochs_eeg, event_ids)

        if rebalance:
            x_eeg, y_eeg = rebalance_classes(x_eeg, y)
            x_pupil, y_pupil = rebalance_classes(x_pupil, y)
            assert np.all(y_eeg == y_pupil)
            y = y_eeg

        if plots == 'sanity_check':
            sanity_check_eeg(x_eeg, y, picks)
            sanity_check_pupil(x_pupil, y)
        elif plots == 'full':
            visualize_eeg_epochs(epochs_eeg, event_ids, colors, title='EEG Epochs ' + title)
            visualize_pupil_epochs(epochs_pupil, event_ids, colors, title='Pupil Epochs ' + title)

        return [x_eeg, x_pupil], y, [epochs_eeg, epochs_pupil], event_ids
    else:
        if data_type == 'eeg':
            tmin = tmin_eeg
            tmax = tmax_eeg
            epochs, event_ids, _, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session, n_jobs=n_jobs, reject=reject, force_square=force_square)
        elif data_type == 'pupil':
            tmin = tmin_pupil
            tmax = tmax_pupil
            epochs, event_ids, ps_group_eeg = rdf.get_pupil_epochs(event_names, event_filters, eyetracking_resample_srate, tmin=tmin_pupil, tmax=tmax_pupil, participant=participant, session=session, n_jobs=n_jobs, force_square=force_square)
        else:
            raise NotImplementedError(f'data type {data_type} is not implemented')
        if force_square:
            epochs = force_square_epochs(epochs, tmin, tmax)
        x, y = _epoch_to_samples(epochs, event_ids)
        # x = []
        # y = []
        # for event_name, event_class in event_ids.items():
        #     x.append(epochs[event_name].get_data(picks=picks))
        #     y += [event_class] * len(epochs[event_name].get_data())
        # x = np.concatenate(x, axis=0)

        if rebalance:
            x, y = rebalance_classes(x, y)

        # x = (x - np.mean(x, axis=(0, 2), keepdims=True)) / np.std(x, axis=(0, 2), keepdims=True)  # z normalize x

        if data_type == 'eeg':
            sanity_check_eeg(x, y, picks)
            if plots == 'sanity_check':
                sanity_check_eeg(x, y, picks)
            elif plots == 'full':
                visualize_eeg_epochs(epochs, event_ids, colors, title='EEG Epochs ' + title)
        elif data_type == 'pupil':
            sanity_check_pupil(x, y)
            if plots == 'sanity_check':
                sanity_check_pupil(x, y)
            elif plots == 'full':
                visualize_pupil_epochs(epochs, event_ids, colors, title='Pupil Epochs ' + title)

        # return x, y, epochs, event_ids, ps_group_eeg
        return x, y, epochs, event_ids


def sanity_check_eeg(x, y, picks):
    coi = picks.index('CPz') if picks else eeg_montage.ch_names.index('CPz')
    x_distractors = x[:, coi, :][y == 0]
    x_targets = x[:, coi, :][y == 1]
    x_distractors = np.mean(x_distractors, axis=0)
    x_targets = np.mean(x_targets, axis=0)
    plt.plot(x_distractors, label=f'distractor, n={np.sum(y == 0)}')
    plt.plot(x_targets, label=f'target, n={np.sum(y == 1)}')
    plt.title('EEG sample sanity check')
    plt.legend()
    plt.show()


def sanity_check_pupil(x, y):
    x_distractors = x[y == 0]
    x_targets = x[y == 1]
    x_distractors = np.mean(x_distractors, axis=(0, 1))  # also average left and right
    x_targets = np.mean(x_targets, axis=(0, 1))

    plt.plot(x_distractors, label=f'distractor, n={np.sum(y == 0)}')
    plt.plot(x_targets, label=f'target, n={np.sum(y == 1)}')
    plt.title('Pupil sample sanity check')
    plt.legend()
    plt.show()

def binary_classification_metric(y_true, y_pred):
    acc = np.sum(y_pred == y_true).item() / len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    FP = np.sum(cm, axis=0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    return acc, TPR, TNR

def z_norm_by_trial(data):
    """
    Z-normalize data by trial, the input data is in the shape of (num_samples, num_channels, num_timesteps)
    @param data: data is in the shape of (num_samples, num_channels, num_timesteps)
    """
    norm_data = np.copy(data)
    for i in range(data.shape[0]):
        sample = data[i]
        mean = np.mean(sample, axis=(0, 1))
        std = np.std(sample, axis=(0, 1))
        sample_norm = (sample - mean) / std
        norm_data[i] = sample_norm
    return norm_data


import numpy as np


def mean_ignore_zero(arr, axis=0):
    """
    Calculates the mean of an array along the specified axis, ignoring all zero values.

    Args:
    - arr: numpy array
    - axis: int (default: 0)

    Returns:
    - mean: numpy array containing the mean of non-zero values along the specified axis
    """
    # Calculate the mean of non-zero values along the specified axis
    mean = np.true_divide(arr.sum(axis=axis), (arr != 0).sum(axis=axis))

    return mean