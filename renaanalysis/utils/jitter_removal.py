import numpy as np


def jitter_removal(timestamps, method='linear fit'):
    coefs = np.polyfit(list(range(len(timestamps))), timestamps, 1)
    smoothed_ts_array = np.array([i * coefs[0] + coefs[1] for i in range(len(timestamps))])
    return smoothed_ts_array
