import pickle

import numpy as np
from rena.utils.data_utils import RNStream

data_path = 'C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/Subjects/0/0.dats'

# data = pickle.load(open(data_path, 'rb'))

stream = RNStream(data_path)
data = stream.stream_in(ignore_stream=('monitor1'), jitter_removal=False)

a = data['Unity.VarjoEyeTrackingComplete'][0]

insert_array = np.zeros(len(data['Unity.VarjoEyeTrackingComplete'][1]))

b = np.insert(a, 6, insert_array, 0)
data['Unity.VarjoEyeTrackingComplete'][0] = b

pickle.dump(data, open(data_path, 'wb'))