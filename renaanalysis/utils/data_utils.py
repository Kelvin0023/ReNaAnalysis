# analysis parameters ######################################################################################
from imblearn.over_sampling import SMOTE
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

from renaanalysis.params.params import *
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
