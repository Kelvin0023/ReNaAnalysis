from collections import defaultdict
from scipy import stats
import pandas as pd
import numpy as np
import scipy.io

SubjNum = pd.read_csv(r'E:\SubjNum.csv',header = None)
Subj_array = SubjNum.to_numpy().flatten() ##Use if the for connection disconnects

condition_time_filtered = pd.read_csv(r'E:\condition_time_filtered.csv',header = None)

# Do 80/20 split for training and testing dataset according to subjects
train_y = np.random.choice(Subj_array, size = int(np.round(0.8*len(Subj_array))), replace = False)
test_y = np.array(list(set(Subj_array) - set(train_y)))

X = condition_time_filtered.drop(columns = ['trialType','RT','subject'])
y = condition_time_filtered['trialType']

# Get index for subjects in training and testing dataset
train_indx = condition_time_filtered.index[condition_time_filtered['subject'].isin(train_y)].tolist()
test_indx = condition_time_filtered.index[condition_time_filtered['subject'].isin(test_y)].tolist()

# Get data from fmri object and trial type numpy array with indices
X_train, X_test = X.iloc[train_indx], X.iloc[test_indx]
y_train, y_test = y.iloc[train_indx], y.iloc[test_indx]

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt

# Split chunks into subjects chunks for cross validation
chunks = np.array(condition_time_filtered['subject'])

# Logistic Regression
print("Start classification")
start_time = time.time()
clf = LogisticRegression(max_iter = 1000)
print("Fitting values")

clf.fit(X_train, y_train.values.ravel())
y_predicted= clf.predict(X_test)
# LogReg_accuracy = clf_dvc.score(X_test, y_test.values.ravel())

print("Logistic Regression Accuracy: " + str(clf.score(X_test, y_test.values.ravel())))
print("--- %s seconds ---" % (time.time() - start_time))

#coefs  = np.array(clf_dvc.coef_).flatten()
#weights = np.insert(coefs, 0, np.array(clf_dvc.intercept_).flatten())
#np.savetxt("dvc_weights.csv", weights, delimiter=",")

print("Start permutation")
start_time = time.time()
score, permutation_score, pval = permutation_test_score(clf,
                                                        X,
                                                        y,
                                                        n_permutations = 1000,
                                                        scoring = 'roc_auc',
                                                        cv = 10,
                                                        groups = chunks)

print("--- %s seconds ---" % (time.time() - start_time))

fig,ax = plt.subplots()
ax.hist(permutation_score, label = 'randomized scores', color = "red")
ax.axvline(score, label = 'true score : %.2f , pval : %.3f'%(score, pval))
ax.legend()
ax.set(title = "Histogram of Permutated Data for" + Type_1_name + " vs. " + Type_2_name, xlabel = "Accuracy Score",
       ylabel = "Count / Frequency of Permutation Scores")
#plt.savefig(Subj+ '_wholeBrain_DecisionTree_permutation.png')


print("ROC AUC: " + str(score))
print("pval : " + str(pval))