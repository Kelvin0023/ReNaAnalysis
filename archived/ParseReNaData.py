import json

from utils.utils import RNStream

session_log_path = 'C:/Recordings/ReNaSessionLog_01-28-2022-21-51-31.json'
itemCatalog_path = 'C:/Recordings/ReNaItemCatalog_01-28-2022-21-51-31.json'
data_path = 'C:/Recordings/01_28_2022_21_52_10-Exp_ReNaTimestampTest-Sbj_ZL-Ssn_0.dats'

session_log = json.load(open(session_log_path, 'r'))
itemCatalog_log = json.load(open(itemCatalog_path, 'r'))

data = RNStream(data_path).stream_in(ignore_stream=('monitor1',), jitter_removal=False)