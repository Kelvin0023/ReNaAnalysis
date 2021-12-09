import json

from rena.utils.data_utils import RNStream

from utils import plot_epochs

tmin = -0.1
tmax = 3
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}


# second participant
data_paths = ['C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/11_13_2021_11_04_11-Exp_ReNaPilot-Sbj_AN-Ssn_0.dats',
              'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/11_10_2021_12_06_17-Exp_ReNaPilot-Sbj_ZL-Ssn_0.dats',]

item_catalog_paths = ['C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/ReNaItemCatalog_11-13-2021-11-03-54.json',
                      'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/ReNaItemCatalog_11-10-2021-12-04-46.json',]

session_log_paths =['C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-13-2021/ReNaSessionLog_11-13-2021-11-03-54.json',
                    'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2021Fall/11-10-2021/ReNaSessionLog_11-10-2021-12-04-46.json']










varjoEyetracking_preset_path = 'D:/PycharmProjects/RealityNavigation/Presets/LSLPresets/VarjoEyeData.json'

title = ''

# process code after this
data = RNStream(data_path).stream_in(ignore_stream=('monitor1'), jitter_removal=False)
item_catalog = json.load(open(item_catalog_path))
session_log = json.load(open(session_log_path))
varjoEyetracking_preset = json.load(open(varjoEyetracking_preset_path))
item_codes = list(item_catalog.values())

# process data
event_markers_rsvp = data['Unity.ReNa.EventMarkers'][0][0:4]
event_markers_carousel = data['Unity.ReNa.EventMarkers'][0][4:8]
event_markers_vs = data['Unity.ReNa.EventMarkers'][0][8:12]

event_markers_timestamps = data['Unity.ReNa.EventMarkers'][1]

itemMarkers = data['Unity.ReNa.ItemMarkers'][0]
itemMarkers_timestamps = data['Unity.ReNa.ItemMarkers'][1]

eyetracking_data = data['Unity.VarjoEyeTracking'][0]
eyetracking_data_timestamps = data['Unity.VarjoEyeTracking'][1]

epochs_rsvp = plot_epochs(event_markers_rsvp, event_markers_timestamps, eyetracking_data, eyetracking_data_timestamps,
                          varjoEyetracking_preset['ChannelNames'], session_log,
                          item_codes, tmin, tmax, color_dict, title=title + ' RSVP')

# epochs_carousel = plot_epochs(event_markers_carousel, event_markers_timestamps, eyetracking_data,
#                               eyetracking_data_timestamps,
#                               varjoEyetracking_preset['ChannelNames'], session_log,
#                               item_codes, tmin, tmax, color_dict, title=title + ' Carousel')
