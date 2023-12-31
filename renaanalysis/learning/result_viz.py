import numpy as np
from matplotlib import pyplot as plt


def viz_performances_rena(metric, results, model_names, conditions_names, lockings, constrained_lockings, constrained_conditions, width=0.175):
    plt.rcParams["figure.figsize"] = (30, 12)
    plt.rcParams.update({'font.size': 18})
    for c in conditions_names:
        this_lockings = lockings if c not in constrained_conditions else constrained_lockings
        ind = np.arange(len(this_lockings))

        m: str
        for m_index, m in enumerate(model_names):
            metric_values = [results[(f"{c}-{l}", m)][metric] for l in this_lockings]  # get the auc for each locking

            plt.bar(ind + m_index * width, metric_values, width, label=f'{m}')
            for j in range(len(metric_values)):
                plt.text(ind[j] + m_index * width, metric_values[j] + 0.05, str(round(metric_values[j], 3)), horizontalalignment='center',
                         verticalalignment='center')

        plt.ylim(0.0, 1.1)
        plt.ylabel(f'{metric} (averaged across folds)')
        plt.title(f'{c}, {metric}')
        plt.xticks(ind + width / 2, this_lockings)
        plt.legend(loc=4)
        plt.show()

def viz_model_performance(results, metrics, model_names, width = 0.175):
    plt.rcParams["figure.figsize"] = (30, 12)
    plt.rcParams.update({'font.size': 18})

    ind = np.arange(len(model_names))
    for metric_index, metric in enumerate(metrics):
        metric_values = [(results[m][metric] if metric in results[m] else np.nan) for m_index, m in enumerate(model_names)]  # get the metric value for each model

        plt.bar(ind + metric_index * width, metric_values, width, label=f'{metric}')
        for j in range(len(metric_values)):
            if metric_values[j] is not np.nan:
                plt.text(ind[j] + metric_index * width, metric_values[j] + 0.05, str(round(metric_values[j], 3)), horizontalalignment='center', verticalalignment='center')

    plt.ylim(0.0, 1.1)
    plt.ylabel(f'Model performance (averaged across folds)')
    plt.title(f'Auditory oddball')
    plt.xticks(ind + width/2, model_names)
    plt.legend(loc='upper left')
    plt.show()