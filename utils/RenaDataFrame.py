import os

import numpy as np

from params import *
from utils.Event import add_events_to_data, check_contraint_block_counts
from utils.utils import generate_pupil_event_epochs, visualize_pupil_epochs, rescale_merge_exg, \
    generate_eeg_event_epochs, preprocess_session_eeg, visualize_eeg_epochs, validate_get_epoch_args


class RenaDataFrame:
    def __init__(self):
        self.participant_session_dict = {}

    def add_participant_session(self, data, events, participant, session_index, bad_channels, ica_path):
        self.participant_session_dict[(participant, session_index)] = data, events, bad_channels, ica_path

    def preprocess(self):
        for (p, s), (data, events, bad_channels, ica_path) in self.participant_session_dict.items():
            if 'BioSemi'in data.keys():
                print(f"Preprocessing EEG for participant {p}, session {s}")
                eeg_raw, eeg_ica_raw, downsampled_timestamps = preprocess_session_eeg(data['BioSemi'], data['BioSemi'][1], ica_path, bad_channels=bad_channels)
                data['BioSemi'] = {'array_original': data['BioSemi'], 'timestamps_original': data['BioSemi'][1], 'raw': eeg_raw, 'ica': eeg_ica_raw, 'timestamps': downsampled_timestamps}

    def get_data_events(self, participant=None, session=None) -> dict:
        """

        :param participant:
        :param session:
        :return: a dictionary with key being (participant, session), value is (data: dict, events: list of Event)
        """
        if participant is None and session is None:
            rtn = dict([((p, s), (data, events)) for (p, s), (data, events, bad_channels, ica_path) in self.participant_session_dict.items()])
            return rtn

        keys = self.participant_session_dict.keys()
        if type(participant) is str:  # single participant index is given
            keys = [(p, s) for p, s in keys if participant == p]
        elif type(participant) is list:
            keys = [(p, s) for p, s in keys if p in participant]
        elif participant is None:
            pass
        else:
            raise TypeError("Unsupported participant type, must be int or list or None")
        if type(session) is int:  # single participant index is given
            keys = [(p, s) for p, s in keys if session == s]
        elif type(session) is list:
            keys = [(p, s) for p, s in keys if s in session]
        elif session is None:
            pass
        else:
            raise TypeError("Unsupported session type, must be int, list or None")
        rtn = dict([((p, s), (data, events)) for (p, s), (data, events, _, _) in self.participant_session_dict.items() if (p, s) in keys])
        return rtn


    def get_pupil_epochs(self, event_names, event_filters, participant=None, session=None):
        """
        event_filters:
        @param event_filters: list of callables, each corresponding to the event name
        @param participant:
        @param session:
        @return:
        """
        validate_get_epoch_args(event_names, event_filters)
        ps_dict = self.get_data_events(participant, session)
        pupil_epochs = None  # clear epochs
        event_ids = None
        for (p, s), (data, events) in ps_dict.items():
            print('Getting pupil epochs for participant {} session {}'.format(p, s))
            eye_data = data[varjoEyetracking_preset["StreamName"]][0]
            pupil_left_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('left_pupil_size'), :]
            pupil_right_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('right_pupil_size'), :]
            pupil_data = np.concatenate([np.expand_dims(pupil_left_data, axis=1), np.expand_dims(pupil_right_data, axis=1)], axis=1)

            pupil_data_with_events, event_ids, deviant = add_events_to_data(pupil_data, data[varjoEyetracking_preset["StreamName"]][1], events, event_names, event_filters)
            epochs_pupil, _ = generate_pupil_event_epochs(pupil_data_with_events, ['pupil_left', 'pupil_right', 'stim'], ['misc', 'misc', 'stim'], event_ids)
            # check_contraint_block_counts(events, deviant + len(epochs_pupil))  # TODO only taken into account constraint conditions
            pupil_epochs = epochs_pupil if pupil_epochs is None else mne.concatenate_epochs([epochs_pupil, pupil_epochs])
        return pupil_epochs, event_ids

    def get_eeg_epochs(self, event_names, event_filters, participant=None, session=None):
        validate_get_epoch_args(event_names, event_filters)
        ps_dict = self.get_data_events(participant, session)
        eeg_epochs = None  # clear epochs
        event_ids = None

        for (p, s), (data, events) in ps_dict.items():
            print('Getting EEG epochs for participant {} session {}'.format(p, s))

            eeg_data_with_events, event_ids, deviant = add_events_to_data(data['BioSemi']['raw'], data['BioSemi']['timestamps'], events, event_names, event_filters)

            epochs, _ = generate_eeg_event_epochs(eeg_data_with_events, event_ids)
            # check_contraint_block_counts(events, deviant + len(epochs))  # TODO only taken into account constraint conditions
            eeg_epochs = epochs if eeg_epochs is None else mne.concatenate_epochs([epochs, eeg_epochs])
        return eeg_epochs, event_ids

    def event_discriminant_analysis(self, event_names, event_filters):

        # TODO
        pass