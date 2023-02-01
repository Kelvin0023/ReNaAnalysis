# analysis parameters ######################################################################################
import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

from renaanalysis.params.params import *
from renaanalysis.params.params import eeg_montage
from renaanalysis.utils.utils import rescale_merge_exg


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


def compute_pca_ica(X, n_components):
    print("applying pca followed by ica")
    # ev = mne.EvokedArray(np.mean(X, axis=0),
    #                      mne.create_info(64, exg_resample_srate,
    #                                      ch_types='eeg'), tmin=-0.1)
    # ev.plot(window_title="original", time_unit='s')
    pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
    pca_data = pca.fit_transform(X)
    # ev = mne.EvokedArray(np.mean(pca_data, axis=0),
    #                      mne.create_info(n_components, exg_resample_srate,
    #                                      ch_types='eeg'), tmin=-0.1)
    # ev.plot( window_title="PCA", time_unit='s')
    ica = UnsupervisedSpatialFilter(FastICA(n_components, whiten='unit-variance'), average=False)
    ica_data = ica.fit_transform(pca_data)

    # ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),mne.create_info(n_components, exg_resample_srate,ch_types='eeg'), tmin=-0.1)
    # ev1.plot(window_title='ICA', time_unit='s')

    return ica_data

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
    sm = SMOTE(random_state=42)
    x, y = sm.fit_resample(x, y)
    x = np.reshape(x, newshape=(len(x),) + epoch_shape)  # reshape back x after resampling
    return x, y


def epochs_to_class_samples(rdf, event_names, event_filters, rebalance=False, participant=None, session=None, picks=None, data_type='eeg', tmin_eeg=-0.1, tmax_eeg=0.8):
    """
    script will always z norm along channels for the input
    @param: data_type: can be eeg, pupil or mixed
    """
    if data_type == 'both':
        epochs_eeg, event_ids, ar_log, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session)
        epochs_pupil, event_ids, ps_group_pupil = rdf.get_pupil_epochs(event_names, event_filters, participant=participant, session=session)
        epochs_pupil = epochs_pupil[np.logical_not(ar_log.bad_epochs)]
        ps_group_pupil = np.array(ps_group_pupil)[np.logical_not(ar_log.bad_epochs)]
        assert np.all(ps_group_pupil == ps_group_eeg)
        y = []
        x_eeg = [epochs_eeg[event_name].get_data(picks=picks) for event_name, _ in event_ids.items()]
        x_pupil = [epochs_pupil[event_name].get_data(picks=picks) for event_name, _ in event_ids.items()]
        x_eeg = np.concatenate(x_eeg, axis=0)
        x_pupil = np.concatenate(x_pupil, axis=0)

        for event_name, event_class in event_ids.items():
            y += [event_class] * len(epochs_pupil[event_name].get_data())
        if np.min(y) == 1:
            y = np.array(y) - 1
        if rebalance:
            x_eeg, y_eeg = rebalance_classes(x_eeg, y)
            x_pupil, y_pupil = rebalance_classes(x_pupil, y)
            assert np.all(y_eeg == y_pupil)
            y = y_eeg
        sanity_check_eeg(x_eeg, y, picks)
        sanity_check_pupil(x_pupil, y)
        return [x_eeg, x_pupil], y, [epochs_eeg, epochs_pupil], event_ids

    if data_type == 'eeg':
        epochs, event_ids, _, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session)
    elif data_type == 'pupil':
        epochs, event_ids, ps_group_eeg = rdf.get_pupil_epochs(event_names, event_filters, participant=participant, session=session)
    else:
        raise NotImplementedError(f'data type {data_type} is not implemented')

    x = []
    y = []
    for event_name, event_class in event_ids.items():
        x.append(epochs[event_name].get_data(picks=picks))
        y += [event_class] * len(epochs[event_name].get_data())
    x = np.concatenate(x, axis=0)

    if np.min(y) == 1:
        y = np.array(y) - 1

    if rebalance:
        x, y = rebalance_classes(x, y)

    x = (x - np.mean(x, axis=(0, 2), keepdims=True)) / np.std(x, axis=(0, 2), keepdims=True)  # z normalize x

    if data_type == 'eeg':
        sanity_check_eeg(x, y, picks)
    elif data_type == 'pupil':
        sanity_check_pupil(x, y)

    # return x, y, epochs, event_ids, ps_group_eeg
    return x, y, epochs, event_ids


def sanity_check_eeg(x, y, picks):
    coi = picks.index('CPz') if picks else eeg_montage.ch_names.index('CPz')
    x_distractors = x[:, coi, :][y == 0]
    x_targets = x[:, coi, :][y == 1]
    x_distractors = np.mean(x_distractors, axis=0)
    x_targets = np.mean(x_targets, axis=0)
    plt.plot(x_distractors, label='distractor')
    plt.plot(x_targets, label='target')
    plt.title('EEG sample sanity check')
    plt.legend()
    plt.show()


def sanity_check_pupil(x, y):
    x_distractors = x[y == 0]
    x_targets = x[y == 1]
    x_distractors = np.mean(x_distractors, axis=(0, 1))  # also average left and right
    x_targets = np.mean(x_targets, axis=(0, 1))

    plt.plot(x_distractors, label='distractor')
    plt.plot(x_targets, label='target')
    plt.title('Pupil sample sanity check')
    plt.legend()
    plt.show()