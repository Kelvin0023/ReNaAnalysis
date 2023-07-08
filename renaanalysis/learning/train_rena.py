import os
import pickle

import mne
import numpy as np

from renaanalysis.learning.train import eval_model
from renaanalysis.params.params import export_data_root
from renaanalysis.utils.rdf_utils import rena_epochs_to_class_samples_rdf


def eval_lockings(rdf, event_names, locking_name_filters, model_name, exg_resample_rate=200, participant=None, session=None, regenerate_epochs=True, n_folds=10, ht_lr=1e-3, ht_l2=1e-6, ht_output_mode='single'):
    # verify number of event types
    eeg_montage = mne.channels.make_standard_montage('biosemi64')
    assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))
    locking_performance = {}
    if participant is None:
        participant = 'all'
    if session is None:
        session = 'all'

    for locking_name, locking_filter in locking_name_filters.items():
        test_name = f'Locking-{locking_name}_P-{participant}_S-{session}'
        if regenerate_epochs:
            x, y, _, _ = rena_epochs_to_class_samples_rdf(rdf, event_names, locking_filter, data_type='both', rebalance=False, participant=participant, session=session, plots='full', exg_resample_rate=exg_resample_rate)
            pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
            pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
        else:
            try:
                x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
                y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
            except FileNotFoundError:
                raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}, locking {locking_name}" + ", EEGPupil" if model_name == 'EEGPupil' else "")
        model_performance, training_histories = eval_model(x[0], x[1], y, event_names, model_name, eeg_montage, test_name=test_name, n_folds=n_folds, exg_resample_rate=exg_resample_rate, ht_lr=ht_lr, ht_l2=ht_l2, ht_output_mode=ht_output_mode)
        for _m_name, _performance in model_performance.items():  # HDCA expands into three models, thus having three performance results
            locking_performance[locking_name, _m_name] = _performance
    return locking_performance
