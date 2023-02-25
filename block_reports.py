import pickle
import numpy as np
import matplotlib.pyplot as plt

lockings = ['VS-FLGI', 'VS-I-VT', 'VS-I-VT-Head', 'VS-Patch-Sim']
metrics = ['accuracy', 'target sensitivity', 'target specificity']
file_name = "D:/PycharmProjects/RenaLabApp/rena/02_22_2023_14_38_01_block_report"
x_bar_offset = 0.2

plt.rcParams.update({'font.size': 30})


block_reports = pickle.load(open(file_name, 'rb'))
num_blocks = int(np.max([block_id for (metablock_num, block_id, locking_name), result_dict in block_reports.items()]))
X_axis = np.arange(num_blocks)

resutls = []
for this_locking in lockings:
    locking_results = dict([(block_id, result_dict) for (metablock_num, block_id, locking_name), result_dict in block_reports.items() if locking_name == this_locking])
    plt.figure(figsize=(24, 10))
    for metric_i, metric in enumerate(metrics):
        metrics_results= np.zeros(num_blocks)
        for i in range(num_blocks):
            if i in locking_results.keys():
                metrics_results[i] = locking_results[i][metric]
        plt.bar(X_axis + x_bar_offset * metric_i, metrics_results, 0.2, label=metric)

    plt.xticks(X_axis + x_bar_offset * len(metrics) / 2, list(range(num_blocks)))
    plt.xlabel("Blocks")
    plt.ylabel("Metrics")
    plt.title(f"Performance of real-time target identification for locking {this_locking}")
    plt.legend()
    plt.show()