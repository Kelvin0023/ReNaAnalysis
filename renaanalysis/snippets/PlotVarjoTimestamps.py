import json
import matplotlib.pyplot as plt
from rena.utils.data_utils import RNStream


tmin = -0.1
tmax = 3
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

trial_data_export_root = 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/SingleTrials'

participant_data_dict = {
                # 'AN': {
                #     'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/12_16_2021_15_40_16-Exp_ReNa-Sbj_AN-Ssn_2.dats',
                #     'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/ReNaItemCatalog_12-16-2021-15-40-01.json',
                #     'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/ReNaSessionLog_12-16-2021-15-40-01.json'},
        #         'ZL': {
        # 'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/12_16_2021_15_40_16-Exp_ReNa-Sbj_AN-Ssn_2.dats',
        # 'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/ReNaItemCatalog_12-16-2021-15-40-01.json',
        # 'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2021Fall/12-16-2021/ReNaSessionLog_12-16-2021-15-40-01.json'},
                    #     'ZL': {
                    # 'data_path': 'C:/Recordings/01_28_2022_22_05_02-Exp_ReNaTimestampTest-Sbj_ZL-Ssn_0.dats',
                    # 'item_catalog_path': 'C:/Recordings/ReNaLogs/ReNaItemCatalog_01-28-2022-22-00-48.json',
                    # 'session_log_path': 'C:/Recordings/ReNaLogs/ReNaSessionLog_01-28-2022-22-00-48.json'}
    'ZL': {
        'data_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/01-31-2022/01_31_2022_15_10_12-Exp_ReNa-Sbj_ZL-Ssn_0.dats',
        'item_catalog_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/01-31-2022/ReNaItemCatalog_01-31-2022-15-09-45.json',
        'session_log_path': 'C:/Users/S-Vec/Dropbox/ReNa/data/ReNaPilot-2022Spring/01-31-2022/ReNaSessionLog_01-31-2022-15-09-45.json'}
}



varjoEyetracking_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'  # TODO this should be VarjoEyeDataComplete
# varjoEyetracking_preset_path = 'C:/Users/LLINC-Lab/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeDataComplete.json'
varjoEyetracking_preset = json.load(open(varjoEyetracking_preset_path))
varjoEyetracking_channelNames = varjoEyetracking_preset['ChannelNames']

title = 'ReNaPilot 2021'
event_ids = {'BlockBegins': 4, 'Novelty': 3, 'Target': 2, 'Distractor': 1}

epochs_pupil_rsvp = None
epochs_pupil_carousel = None

for participant_index, participant_code_data_path_dict in enumerate(participant_data_dict.items()):
    participant_code, participant_data_path_dict = participant_code_data_path_dict

    data_path = participant_data_path_dict['data_path']
    item_catalog_path = participant_data_path_dict['item_catalog_path']
    session_log_path = participant_data_path_dict['session_log_path']
    # process code after this
    data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
    item_catalog = json.load(open(item_catalog_path))
    session_log = json.load(open(session_log_path))

    item_codes = list(item_catalog.values())

    # process data  # TODO iterate over conditions
    event_markers_rsvp = data['Unity.ReNa.EventMarkers'][0][0:4]
    event_markers_carousel = data['Unity.ReNa.EventMarkers'][0][4:8]
    event_markers_vs = data['Unity.ReNa.EventMarkers'][0][8:12]

    event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]

    itemMarkers = data['Unity.ReNa.ItemMarkers'][0]
    itemMarkers_timestamps = data['Unity.ReNa.ItemMarkers'][1]

    eyetracking_data = data['Unity.VarjoEyeTrackingComplete'][0]
    eyetracking_data_timestamps = data['Unity.VarjoEyeTrackingComplete'][1]

    print('Plotting timestamps for participant {0}'.format(participant_code))
    plt.plot(eyetracking_data[0, :])
    plt.xlabel('Timestamp Index')
    plt.ylabel('Varjo Timestamp')
    plt.title('Participant code: {0}'.format(participant_code))
    plt.show()