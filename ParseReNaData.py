import json

import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import RNStream

session_log_path = 'C:/Users/S-Vec/Downloads/ReNa Pilot/ReNaSessionLog_10-30-2021-21-03-24.json'
itemCatalog_path = 'C:/Users/S-Vec/Downloads/ReNa Pilot/ReNaItemCatalog_10-30-2021-21-03-24.json'
data_path = 'C:/Users/S-Vec/Downloads/ReNa Pilot/10_30_2021_21_03_46-Exp_ReNa-Sbj_Pilot1-Ssn_1.dats'

session_log = json.load(open(session_log_path, 'r'))
itemCatalog_log = json.load(open(itemCatalog_path, 'r'))

RNStream