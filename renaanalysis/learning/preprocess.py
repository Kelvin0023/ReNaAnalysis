import os
import pickle
from typing import List

from renaanalysis.params.params import pupil_name, eeg_name, fnirs_name
from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica
from renaanalysis.multimodal.PhysioArray import PhysioArray


def preprocess_samples_eeg_pupil(x_eeg, x_pupil, n_top_components=20):
    x_eeg_znormed = z_norm_by_trial(x_eeg)
    x_pupil_znormed = z_norm_by_trial(x_pupil) if x_pupil is not None else None
    x_eeg_pca_ica, pca, ica = compute_pca_ica(x_eeg, n_top_components)
    return x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica

def preprocess_eeg(x_eeg, n_top_components=20):
    x_eeg_znormed = z_norm_by_trial(x_eeg)
    x_eeg_pca_ica, pca, ica = compute_pca_ica(x_eeg, n_top_components)
    return x_eeg_znormed, x_eeg_pca_ica, pca, ica

def preprocess_samples_and_save(physio_arrays: List[PhysioArray], epochs_root: str, is_apply_pca_ica_eeg, pca_ica_eeg_n_components=20, dataset_name=None):
    """
    For eeg, this function checks if preprocessed data exists, if not, it will preprocess the data and save it.

    Pupil data's preprocess will always compute for it is relatively fast.
    @param physio_arrays:
    @param epochs_root:
    @param dataset_name:
    @param is_apply_pca_ica_eeg:
    @param pca_ica_eeg_n_components:
    @return:
    """

    x_dict_preprocessed = dict()
    for i in range(len(physio_arrays)):
        parray_preprocessed_eeg_file_path = f'{epochs_root}/x_{str(physio_arrays[i])}_preprocessed.p'
        if physio_arrays[i].physio_type == eeg_name:
            if dataset_name == 'BCICIVA':
                physio_arrays[i].apply_znorm_by_run()
            else:
                physio_arrays[i].apply_znorm_by_trial()
            if is_apply_pca_ica_eeg:
                if os.path.exists(parray_preprocessed_eeg_file_path):
                    physio_arrays[i] = pickle.load(open(parray_preprocessed_eeg_file_path, "rb"))
                physio_arrays[i].apply_pca_ica(pca_ica_eeg_n_components)  # this will check if pca_ica_eeg_n_components matches
            pickle.dump(physio_arrays[i], open(parray_preprocessed_eeg_file_path, "wb"))
        elif physio_arrays[i].physio_type == pupil_name:
            physio_arrays[i].apply_znorm_by_trial()
            pickle.dump(physio_arrays[i], open(parray_preprocessed_eeg_file_path, "wb"))
        elif physio_arrays[i].physio_type == fnirs_name:
            physio_arrays[i].apply_znorm_global()
            pickle.dump(physio_arrays[i], open(parray_preprocessed_eeg_file_path, "wb"))
        else:
            raise NotImplementedError

    return physio_arrays

