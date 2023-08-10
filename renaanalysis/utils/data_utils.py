# analysis parameters ######################################################################################
from autoreject import AutoReject
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import confusion_matrix

from renaanalysis.params.params import eeg_montage, eeg_picks
from renaanalysis.utils.utils import rescale_merge_exg, visualize_eeg_epochs


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
    @param X: input array
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

    return (x_train - projection_mean) / projection_std, (x_test - projection_mean) / projection_std, projection_mean, projection_std

def z_norm_hdca(x, _mean=None, _std=None):
    if _mean is None or _std is None:
        _mean = np.mean(x, axis=0, keepdims=True)
        _std = np.std(x, axis=0, keepdims=True)
    return (x - _mean) / _std

def rebalance_classes(x, y, by_channel=False, random_seed=None):
    """
    Resamples the data to balance the classes using SMOTE algorithm.

    Parameters:
        x (np.ndarray): Input data array of shape (epochs, channels, samples).
        y (np.ndarray): Target labels array of shape (epochs,).
        by_channel (bool): If True, balance the classes separately for each channel. Otherwise,
            balance the classes for the whole input data.

    Returns:
        tuple: A tuple containing the resampled input data and target labels as numpy arrays.
    """
    epoch_shape = x.shape[1:]

    if by_channel:
        y_resample = None
        channel_data = []
        channel_num = epoch_shape[0]

        # Loop through each channel and balance the classes separately
        for channel_index in range(0, channel_num):
            sm = SMOTE(random_state=random_seed)
            x_channel = x[:, channel_index, :]
            x_channel, y_resample = sm.fit_resample(x_channel, y)
            channel_data.append(x_channel)

        # Expand dimensions for each channel array and concatenate along the channel axis
        channel_data = [np.expand_dims(x, axis=1) for x in channel_data]
        x = np.concatenate([x for x in channel_data], axis=1)
        y = y_resample

    else:
        # Reshape the input data to 2D array and balance the classes
        x = np.reshape(x, newshape=(len(x), -1))
        sm = SMOTE(random_state=random_seed)
        x, y = sm.fit_resample(x, y)

        # Reshape the input data back to its original shape
        x = np.reshape(x, newshape=(len(x),) + epoch_shape)

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

def _epoch_to_samples(epochs, event_ids, picks=None, perserve_order=True, event_marker_to_label=True, require_metainfo=False):
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
    if require_metainfo:
        metadata = dict([(k, np.array(v)) for k, v in epochs.metadata.items()])
        channel_positions = epochs.get_montage().get_positions()['ch_pos']
        epoch_channel_positions = np.array([channel_positions[ch_name] for ch_name in epochs.ch_names])
        metadata['channel_positions'] = np.repeat(epoch_channel_positions[np.newaxis, :, :], len(epochs), axis=0)
    else:
        metadata = None
    if len(event_ids.keys()) == 1:
        y = None

    return x, y, metadata

def force_square_epochs(epochs, tmin, tmax):
    # get the number of events overall, so we know what the number of time points should be to make the data matrix square
    if epochs.get_data().shape[2] != (num_epochs := epochs.get_data().shape[0]):
        target_resample_srate = num_epochs / (tmax - tmin)
        square_epochs = epochs.resample(target_resample_srate)
    return square_epochs

def epochs_to_class_samples_TUH(epochs, event_names, picks=None,
                            eeg_viz_picks =eeg_picks,
                            eeg_resample_rate=128, n_jobs=1, reject='auto', plots='sanity-check', colors=None, title='', random_seed=None, low_freq=None, high_freq=None, require_metainfo=True,
                            is_plot_ERP=True, is_plot_PSD=True, epoch_tmin=1, epoch_tmax=3, *args, **kwargs):
    """
    script will always z norm along channels for the input
    @param: data_type: can be eeg, pupil or mixed
    @param: force_square: whether to call resample again on the data to force the number of epochs to match the
    number of time points. Enabling this can help algorithms that requires square matrix as their input. Default
    is disabled. Note when force_square is enabled, the resample rate (both eeg and pupil) will be ignored. rebalance
    will also be disabled.
    @param: plots: can be 'sanity_check', 'full', or none
    """
    montage_type_map = {
        '02_tcp_le': 0,
        '01_tcp_ar': 1,
        '03_tcp_ar_a': 2,
        '04_tcp_le_a': 3,
    }
    for idx, montage_type in enumerate(epochs.metadata['montage_type_name']):
        epochs.events[idx, 2] = montage_type_map[montage_type]
    epochs.event_id = montage_type_map
    if picks is not None:
        epochs = epochs.copy().pick(picks)[event_names]
    else:
        epochs = epochs.copy()[event_names]

    if low_freq is not None and high_freq is not None:
        epochs.filter(low_freq, high_freq, n_jobs=n_jobs)
    if eeg_resample_rate is not None and epochs.info['sfreq'] != eeg_resample_rate:
        epochs.resample(eeg_resample_rate, n_jobs=n_jobs)
    if reject == 'auto':
        print("Auto rejecting epochs")
        ar = AutoReject(n_jobs=n_jobs, verbose=False, random_state=random_seed, *args, **kwargs)
        epochs_clean, log = ar.fit_transform(epochs, return_log=True)
    else:
        epochs_clean = epochs
    x, y, metadata = _epoch_to_samples(epochs_clean, montage_type_map, require_metainfo=require_metainfo, event_marker_to_label=False)
    visualize_eeg_epochs(epochs_clean, montage_type_map, colors, tmin_eeg_viz=epoch_tmin, tmax_eeg_viz=epoch_tmax, eeg_picks=eeg_viz_picks, title='EEG Epochs ' + title, low_frequency=low_freq, high_frequency=high_freq, is_plot_PSD=is_plot_PSD, is_plot_timeseries=is_plot_ERP)

    return x, y, metadata

def epochs_to_class_samples(epochs, event_names, picks=None,
                            eeg_viz_picks =eeg_picks,
                            eeg_resample_rate=128, n_jobs=1, reject='auto', plots='sanity-check', colors=None, title='', random_seed=None, low_freq=None, high_freq=None, require_metainfo=True,
                            is_plot_ERP=True, is_plot_PSD=True, epoch_tmin=1, epoch_tmax=3, *args, **kwargs):
    """
    script will always z norm along channels for the input
    @param: data_type: can be eeg, pupil or mixed
    @param: force_square: whether to call resample again on the data to force the number of epochs to match the
    number of time points. Enabling this can help algorithms that requires square matrix as their input. Default
    is disabled. Note when force_square is enabled, the resample rate (both eeg and pupil) will be ignored. rebalance
    will also be disabled.
    @param: plots: can be 'sanity_check', 'full', or none
    """
    if picks is not None:
        epochs = epochs.copy().pick(picks)[event_names]
    else:
        epochs = epochs.copy()[event_names]

    if low_freq is not None and high_freq is not None:
        epochs.filter(low_freq, high_freq, n_jobs=n_jobs)
    if eeg_resample_rate is not None and epochs.info['sfreq'] != eeg_resample_rate:
        epochs.resample(eeg_resample_rate, n_jobs=n_jobs)
    if reject == 'auto':
        print("Auto rejecting epochs")
        ar = AutoReject(n_jobs=n_jobs, verbose=False, random_state=random_seed, *args, **kwargs)
        epochs_clean, log = ar.fit_transform(epochs, return_log=True)
    else:
        epochs_clean = epochs
    event_ids = {event_name: i for i, event_name in enumerate(event_names)}
    x, y, metadata = _epoch_to_samples(epochs_clean, event_ids, require_metainfo=require_metainfo, event_marker_to_label=False)
    visualize_eeg_epochs(epochs_clean, event_ids, colors, tmin_eeg_viz=epoch_tmin, tmax_eeg_viz=epoch_tmax, eeg_picks=eeg_viz_picks, title='EEG Epochs ' + title, low_frequency=low_freq, high_frequency=high_freq, is_plot_PSD=is_plot_PSD, is_plot_timeseries=is_plot_ERP)

    return x, y, metadata



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

def min_max_by_trial(data):
    """
    Min-max normalize data by trial, the input data is in the shape of (num_samples, num_channels, num_timesteps)
    @param data: data is in the shape of (num_samples, num_channels, num_timesteps)
    """
    norm_data = np.copy(data)
    for i in range(data.shape[0]):
        sample = data[i]
        min = np.min(sample, axis=(0, 1))
        max = np.max(sample, axis=(0, 1))
        sample_norm = (sample - min) / (max - min)
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


def check_and_merge_dicts(dict_dict):
    encountered_keys = set()
    merged_dict = {}

    for _key, d in dict_dict.items():
        for key in d:
            # this_key = _key + '_' + key
            this_key = key
            merged_dict[this_key] = d[key]
            if this_key in encountered_keys:
                raise ValueError(f"Duplicate key found: {this_key}")
            encountered_keys.add(this_key)

    return merged_dict


def check_arrays_equal(array_list):
    reference_array = array_list[0]
    for arr in array_list[1:]:
        if not np.array_equal(reference_array, arr):
            return False
    return True