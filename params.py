event_ids_dict = {'EventMarker': {'DistractorPops': 1, 'TargetPops': 2, 'NoveltyPops': 3},
            'GazeRayIntersect': {'GazeRayIntersectsDistractor': 4, 'GazeRayIntersectsTarget': 5, 'GazeRayIntersectsNovelty': 6},
            'GazeBehavior': {'FixationDistractor': 7, 'FixationTarget': 8, 'FixationNovelty': 9, 'FixationNull': 10,
                              'Saccade2Distractor': 11, 'Saccade2Target': 12, 'Saccade2Novelty': 13,
                              'Saccade2Null': 14},
                  }  # event_ids_for_interested_epochs
color_dict = {
              'DistractorPops': 'blue', 'TargetPops': 'red', 'NoveltyPops': 'orange',
              'Fixation': 'blue', 'Saccade': 'orange',
                "GazeRayIntersectsDistractor": 'blue', "GazeRayIntersectsTarget": 'red', "GazeRayIntersectsNovelty": 'orange',
              'FixationDistractor': 'blue', 'FixationTarget': 'red', 'FixationNovelty': 'orange', 'FixationNull': 'grey',
              'Saccade2Distractor': 'blue', 'Saccade2Target': 'red', 'Saccade2Novelty': 'orange', 'Saccade2Null': 'yellow'}

event_viz = 'GazeRayIntersect'


conditions = ['RSVP', 'Carousel']

base_root = "C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/"
data_directory = "Subjects"
varjoEyetrackingComplete_preset_path = 'presets/VarjoEyeDataComplete.json'
eventmarker_preset_path = 'presets/ReNaEventMarker.json'

tmin_pupil = -1
tmax_pupil = 3.
tmin_pupil_viz = -0.1
tmax_pupil_viz = 3.

tmin_eeg = -1.2
tmax_eeg = 2.4

tmin_eeg_viz = -0.1
tmax_eeg_viz = 1.2

eyetracking_srate = 200
exg_srate = 2048

eeg_picks = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']

eeg_channel_names = mne.channels.make_standard_montage('biosemi64').ch_names
ecg_ch_name='ECG00'

note = "test_v3"