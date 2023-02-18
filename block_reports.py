import pickle
import numpy as np
import matplotlib.pyplot as plt

file_name = "C:/Users/S-Vec/Downloads/02_16_2023_20_13_33_block_report"

block_reports = pickle.load(open(file_name, 'rb'))

resutls = []
for (_, block_num, locking_name), result_dict in block_reports.items():
    if locking_name == 'VS-FLGI':
        resutls.append(result_dict['target sensitivity'])