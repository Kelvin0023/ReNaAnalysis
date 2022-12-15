import warnings

from autoreject import AutoReject

from params import *
from utils.Event import add_events_to_data
from utils.utils import generate_pupil_event_epochs, generate_eeg_event_epochs, preprocess_session_eeg, \
    validate_get_epoch_args, \
    interpolate_zeros


class RenaDataFrame:
    def __init__(self):
        self.participant_session_videos = {}
        self.participant_session_dict = {}

    def add_participant_session(self, data, events, participant, session_index, bad_channels, ica_path, video_dir):
        self.participant_session_dict[(participant, session_index)] = data, events, bad_channels, ica_path
        self.participant_session_videos[(participant, session_index)] = video_dir

    def preprocess(self):
        for (p, s), (data, events, bad_channels, ica_path) in self.participant_session_dict.items():
            if 'BioSemi'in data.keys():
                print(f"Preprocessing EEG for participant {p}, session {s}")
                eeg_raw, eeg_ica_raw, downsampled_timestamps = preprocess_session_eeg(data['BioSemi'], data['BioSemi'][1], ica_path, bad_channels=bad_channels)
                data['BioSemi'] = {'array_original': data['BioSemi'], 'timestamps_original': data['BioSemi'][1], 'raw': eeg_raw, 'ica': eeg_ica_raw, 'timestamps': downsampled_timestamps}
            if 'Unity.VarjoEyeTrackingComplete' in data.keys():
                print(f"Preprocessing pupil for participant {p}, session {s}")
                left = data['Unity.VarjoEyeTrackingComplete'][0][varjoEyetracking_preset['ChannelNames'].index('left_pupil_size')].copy()
                assert np.sum(left == np.nan) == 0
                left = interpolate_zeros(left)
                data['Unity.VarjoEyeTrackingComplete'][0][varjoEyetracking_preset['ChannelNames'].index('left_pupil_size')] = left

                right = data['Unity.VarjoEyeTrackingComplete'][0][varjoEyetracking_preset['ChannelNames'].index('right_pupil_size')].copy()
                assert np.sum(right == np.nan) == 0
                right = interpolate_zeros(right)
                data['Unity.VarjoEyeTrackingComplete'][0][varjoEyetracking_preset['ChannelNames'].index('right_pupil_size')] = right

    def get_data_events(self, participant=None, session=None) -> dict:
        """

        :param participant:
        :param session:
        :return: a dictionary with key being (participant, session), value is (data: dict, events: list of Event)
        """
        if participant is None and session is None:
            rtn = dict([((p, s), (data, events)) for (p, s), (data, events, bad_channels, ica_path) in self.participant_session_dict.items()])
            return rtn

        keys = self.get_filtered_particiapnt_session_key(participant, session)
        rtn = dict([((p, s), (data, events)) for (p, s), (data, events, _, _) in self.participant_session_dict.items() if (p, s) in keys])
        return rtn

    def get_event(self, participant, session):
        try:
            assert type(participant) is str and type(session) is int
        except AssertionError:
            raise TypeError(f"Wrong type for participant {type(participant)} or session {type(session)}")
        keys = self.get_filtered_particiapnt_session_key(participant, session)
        return self.participant_session_dict[keys[0]][1]

    def get_filtered_particiapnt_session_key(self, participant=None, session=None):
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
        return keys
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
        ps_group = []

        for (i, (p, s), (data, events)) in enumerate(ps_dict.items()):
            print('Getting pupil epochs for participant {} session {}'.format(p, s))
            eye_data = data[varjoEyetracking_preset["StreamName"]][0]
            pupil_left_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('left_pupil_size'), :]
            pupil_right_data = eye_data[varjoEyetracking_preset["ChannelNames"].index('right_pupil_size'), :]
            pupil_data = np.concatenate([np.expand_dims(pupil_left_data, axis=1), np.expand_dims(pupil_right_data, axis=1)], axis=1)

            pupil_data_with_events, event_ids, deviant = add_events_to_data(pupil_data, data[varjoEyetracking_preset["StreamName"]][1], events, event_names, event_filters)
            epochs_pupil, _ = generate_pupil_event_epochs(pupil_data_with_events, ['pupil_left', 'pupil_right', 'stim'], ['misc', 'misc', 'stim'], event_ids)
            # check_contraint_block_counts(events, deviant + len(epochs_pupil))  # TODO only taken into account constraint conditions
            if len(epochs_pupil) == 0:
                warnings.warn(f'No epochs found for participant {p} session {s} after rejection, skipping')
            else:
                print(f'Found {len(epochs_pupil)} pupil epochs for participant {p} session {s}')
            pupil_epochs = epochs_pupil if pupil_epochs is None else mne.concatenate_epochs([epochs_pupil, pupil_epochs])
            ps_group += [i] * len(pupil_epochs)

        return pupil_epochs, event_ids, ps_group

    def get_eeg_epochs(self, event_names, event_filters, tmin, tmax, participant=None, session=None):
        validate_get_epoch_args(event_names, event_filters)
        ps_dict = self.get_data_events(participant, session)
        eeg_epochs = None  # clear epochs
        event_ids = None
        ps_group = []
        for (i, (p, s), (data, events)) in enumerate(ps_dict.items()):
            eeg_data_with_events, event_ids, deviant = add_events_to_data(data['BioSemi']['raw'], data['BioSemi']['timestamps'], events, event_names, event_filters)

            epochs, _ = generate_eeg_event_epochs(eeg_data_with_events, event_ids, tmin, tmax)
            # check_contraint_block_counts(events, deviant + len(epochs))  # TODO only taken into account constraint conditions
            if len(epochs) == 0:
                warnings.warn(f'No epochs found for participant {p} session {s} after rejection, skipping')
            else:
                print(f'Found {len(epochs)} EEG epochs for participant {p} session {s}')
                eeg_epochs = epochs if eeg_epochs is None else mne.concatenate_epochs([epochs, eeg_epochs])
            ps_group += [i] * len(eeg_epochs)
        print("Auto rejecting epochs")
        ar = AutoReject(n_jobs=20, verbose=False)
        eeg_epochs_clean, log = ar.fit_transform(eeg_epochs, return_log=True)
        return eeg_epochs_clean, event_ids, log, ps_group

