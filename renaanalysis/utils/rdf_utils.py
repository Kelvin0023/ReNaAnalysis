import numpy as np

from renaanalysis.utils.data_utils import force_square_epochs, _epochs_to_samples_eeg_pupil, rebalance_classes, \
    sanity_check_eeg, sanity_check_pupil, _epoch_to_samples
from renaanalysis.utils.utils import visualize_eeg_epochs, visualize_pupil_epochs


def rena_epochs_to_class_samples_rdf(rdf, event_names, event_filters, *, rebalance=False, participant=None, session=None, picks=None, data_type='eeg',
                                     tmin_eeg=-0.1, tmax_eeg=0.8, exg_resample_rate=128,
                                     tmin_pupil=-1., tmax_pupil=3., eyetracking_resample_srate=20, n_jobs=1, reject='auto', force_square=False, plots='sanity-check', colors=None, title=''):
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
        eyetracking_resample_srate = exg_resample_rate = None
        rebalance = False
    if data_type == 'both':
        epochs_eeg, event_ids, ar_log, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session, resample_rate=exg_resample_rate, n_jobs=n_jobs, reject=reject)
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
        x, y, *_ = _epoch_to_samples(epochs, event_ids)
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
