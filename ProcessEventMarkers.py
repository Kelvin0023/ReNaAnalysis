import json
import numpy as np
from src.utils.data_utils import RNStream
import matplotlib.pyplot as plt
import numpy as np

test_rns = RNStream('C:/Recordings/11_10_2021_12_06_17-Exp_ReNa-Sbj_Leo-Ssn_0.dats')
data = test_rns.stream_in(ignore_stream=('monitor1'),jitter_removal=False)

f = open('C:/Recordings/ReNaLogs/ReNaItemCatalog_10-30-2021-21-03-24.json')
item_catalogue = json.load(f)

f = open('C:/Recordings/ReNaLogs/ReNaSessionLog_10-30-2021-21-03-24.json')
session_log = json.load(f)

item_codes = list(item_catalogue.values())

rsvp_event_markers = data['Unity.ReNa.EventMarkers'][0][0]
rsvp_event_type = []
block_num = None

for (i,event) in enumerate(rsvp_event_markers):
    print(event)
    if str(int(event)) in session_log.keys():
        print(event)
        block_num = event
        continue

    if event == -1:
        break;

    if event in item_codes:
        print(event)
        targets = session_log[str(int(block_num))]['targets']
        distractors = session_log[str(int(block_num))]['distractors']
        novelties = session_log[str(int(block_num))]['novelties']
        if event in targets:
            rsvp_event_type.append(2)
        elif ((event in distractors) or (event in novelties)):
            rsvp_event_type.append(1)