import pickle

import numpy as np

data_path = 'C:/Users/LLINC-Lab/Dropbox/ReNa/data/ReNaPilot-2022Fall/Subjects/0/0.p'

data = pickle.load(open(data_path, 'rb'))

a = data['Unity.VarjoEyeTrackingComplete'][0]

insert_array = np.zeros(len(data['Unity.VarjoEyeTrackingComplete'][1]))

data['Unity.VarjoEyeTrackingComplete'][0] = np.insert(a, 5, insert_array, 0)

pickle.dump(data, open(data_path, 'wb'))