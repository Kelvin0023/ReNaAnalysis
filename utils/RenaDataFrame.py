import numpy as np

from params import *
from utils.Event import add_events_to_data, check_contraint_block_counts
from utils.utils import generate_pupil_event_epochs, visualize_pupil_epochs


class RenaDataFrame:
    def __init__(self):
        self.participant_session_dict = {}
        self.pupil_epochs = None

    def add_participant_session(self, data, events, participant, session):
        self.participant_session_dict[(participant, session)] = data, events

    def get_data_events(self, participant=None, session=None):
        if participant is None and session is None:
            return self.participant_session_dict

        keys = self.participant_session_dict.keys()
        if type(participant) == int:  # single participant index is given
            # TODO
            keys = [k for k in keys if participant in k]

    def get_pupil_epochs(self, event_names, event_filters, participant=None, session=None):
        """
        event_filters: list of callables
        @param event_filters:
        @param participant:
        @param session:
        @return:
        """
        ps_dict = self.get_data_events(participant, session)
        self.pupil_epochs = None
        self.pupil_event_ids = None

        for (p, s), (data, events) in ps_dict.items():
            print('Getting pupil epochs for participant {} session {}'.format(p, s))
            eye_data = data[varjoEyetracking_preset["StreamName"]][0]
            pupil_left_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('left_pupil_size'), :]
            pupil_right_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('right_pupil_size'), :]
            pupil_data = np.concatenate([np.expand_dims(pupil_left_data, axis=1), np.expand_dims(pupil_right_data, axis=1)], axis=1)

            pupil_data_with_events, self.pupil_event_ids, deviant = add_events_to_data(pupil_data, data[varjoEyetracking_preset["StreamName"]][1], events, event_names, event_filters)
            epochs_pupil, _ = generate_pupil_event_epochs(pupil_data_with_events, ['pupil_left', 'pupil_right', 'stim'], ['misc', 'misc', 'stim'], self.pupil_event_ids)
            check_contraint_block_counts(events, deviant + len(epochs_pupil))  # TODO only taken into account constraint conditions
            self.pupil_epochs = epochs_pupil if self.pupil_epochs is None else mne.concatenate_epochs([epochs_pupil, self.pupil_epochs])

    def viz_pupil_epochs(self, event_names, event_filters, colors, participant=None, session=None, regen_epochs=False):
        assert len(event_filters) == len(colors)

        if self.pupil_epochs == None or regen_epochs:
            self.get_pupil_epochs(event_names, event_filters, participant, session)

        visualize_pupil_epochs(self.pupil_epochs, self.pupil_event_ids, colors)
        pass
