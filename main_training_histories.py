import pickle

from matplotlib import pyplot as plt

history_file_path = 'results/model_performances_auditory_oddball2023-05-12_12-42-02HT_training_history'
# history_file_path = 'results/model_performances_auditory_oddball2023-05-12_12-42-02EEGCNN_training_history'

training_histories = pickle.load(open(history_file_path, 'rb'))
num_folds = len(training_histories['loss_train'])

for i in range(num_folds):
    plt.plot(training_histories["loss_train"][i], label='train loss')
    plt.plot(training_histories["loss_val"][i], label='val loss')
    plt.title(f'model loss, fold {i}')
    plt.show()

    plt.plot(training_histories["acc_train"][i], label='train accuracy')
    plt.plot(training_histories["acc_val"][i], label='val accuracy')
    plt.title(f'model accuracy, fold {i}')
    plt.show()