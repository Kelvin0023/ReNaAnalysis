import pickle
import numpy as np
import matplotlib.pyplot as plt

lockings = ['VS-FLGI', 'VS-I-DT-Head', 'VS-I-VT-Head', 'VS-Patch-Sim']
metrics = ['accuracy', 'target sensitivity', 'target specificity']
# file_name = "D:/PycharmProjects/RenaLabApp/rena/03_30_2023_14_52_37_block_report"
file_name = "C:/Users/LLINC-Lab/PycharmProjects/RealityNavigation/rena/04_04_2023_21_08_58_block_report"
x_bar_offset = 0.2

plt.rcParams.update({'font.size': 30})


block_reports = pickle.load(open(file_name, 'rb'))
# num_blocks = int(np.max([block_id for (metablock_num, block_id, locking_name), result_dict in block_reports.items()]))
block_ids = np.unique([x[1] for x in block_reports.keys()])
X_axis = np.arange(len(block_ids))

resutls = []
for this_locking in lockings:
    locking_results = dict([(block_id, result_dict) for (metablock_num, block_id, locking_name), result_dict in block_reports.items() if locking_name == this_locking])
    plt.figure(figsize=(24, 10))
    for metric_i, metric in enumerate(metrics):
        metrics_results = np.zeros(len(block_ids))
        for idx, block_id in enumerate(block_ids):
            metrics_results[idx] = locking_results[block_id][metric]
        plt.bar(X_axis + x_bar_offset * metric_i, metrics_results, 0.2, label=metric)

    plt.xticks(X_axis + x_bar_offset * len(metrics) / 2, list(range(len(block_ids))))
    plt.xlabel("Blocks")
    plt.ylabel("Metrics")
    plt.title(f"Performance of real-time target identification for locking {this_locking}")
    plt.legend()
    plt.show()