import numpy as np
from matplotlib import pyplot as plt


def viz_performances(metric, results, model_names, conditions_names, lockings, constrained_lockings, constrained_conditions, width=0.175):
    for c in conditions_names:
        this_lockings = lockings if c not in constrained_conditions else constrained_lockings
        ind = np.arange(len(this_lockings))

        for m_index, m in enumerate(model_names):
            aucs = [results[(f'{c}-{l}', m)]['folds val auc'] for l in this_lockings]  # get the auc for each locking

            plt.bar(ind + m_index * width, aucs, width, label=f'{m}')
            for j in range(len(aucs)):
                plt.text(ind[j] + m_index * width, aucs[j] + 0.05, str(round(aucs[j], 3)), horizontalalignment='center',
                         verticalalignment='center')

        plt.ylim(0.0, 1.1)
        plt.ylabel('AUC (averaged across folds)')
        plt.title(f'Condition {c}, AUC by model and lockings')
        plt.xticks(ind + width / 2, this_lockings)
        plt.legend(loc=4)
        plt.show()