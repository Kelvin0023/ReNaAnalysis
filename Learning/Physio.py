import math
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy import interpolate
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils import window_slice, build_train_rnn, plot_train_history, plot_cm_results, build_train_cnn, build_train_ann, \
    build_train_birnn_with_attention, get_img_from_fig, plot_roc_multiclass
import matplotlib.pyplot as plt


scn = 'RSVP'

X = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL.npy')
y = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL_labels.npy')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3, shuffle=True)

# test the ANN
history_ann = build_train_ann(x_train, x_test, y_train, y_test)
plot_train_history(history_ann, note=str(scn) + ' ANN')
eval = history_ann.model.evaluate(x=x_test, y=y_test)

# # test the RNN
# history_rnn = build_train_rnn(x_train, x_test, y_train, y_test)
# plot_train_history(history_rnn, note=str(scn) + ' RNN')
# eval = history_rnn.model.evaluate(x=x_test, y=y_test)
#
# # test the CNN
# history_cnn = build_train_cnn(x_train, x_test, y_train, y_test)
# plot_train_history(history_cnn, note=str(scn) + ' CNN')
# eval = history_cnn.model.evaluate(x=x_test, y=y_test)

# test the BIRNN_attention
# history_brnn = build_train_birnn_with_attention(x_train, x_test, y_train, y_test)
# plot_train_history(history_brnn, note=str(scn) + ' BIRNN_attention')
# eval = history_brnn.model.evaluate(x=x_test, y=y_test)



# scenario_train_histories_without_model = list([mdl_scn, his_eval[0].history] for mdl_scn, his_eval in scenario_train_histories.items())
# pickle.dump(scenario_train_histories_without_model, open('results/scenario_train_histories_111420.p', 'wb'))
#
# rename_map = {'foo': 'foot', 'han': 'hand', 'hea': 'head'}
# rename_freq_map = {'04g': '0.4 GHz', '24g': '2.4 GHz', '5g': '5 GHz'}
#
# freqs = ['0.4 GHz', '2.4 GHz', '5 GHz']
#
#
# scenario_train_histories_rename = dict()
# for key, value in scenario_train_histories.items():
#     key_rename = (key[0], (rename_map[key[1][0]], rename_freq_map[key[1][1]]))
#     scenario_train_histories_rename[key_rename] = value
#
# # plot scenario-based performance
# width = 0.175
# plt.rcParams["figure.figsize"] = (8, 4)
# SMALL_SIZE = 12
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 16
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)
# # for scns in [locs, freqs]:
# #     strct_sns_acc_dict = dict([(model_strct, []) for model_strct in
# #                                strcts.keys()])  # i.e., ('ANN', [0.9, 0.8, 0.3]) in the list are the average accuracy for this sencario, sencario could be 04g
# #     for sn in scns:  # i.e., sn = 04g, scns = '04g', '24g', '5g', find marginal accuracy for all 04g
# #         for model_strct in strcts.keys():  # iterate through model structures: ANN, RNN, CNN
# #             sn_strct_accuracies = [hist_eval[1][1] for strct_scn, hist_eval in scenario_train_histories_rename.items()
# #                                    if strct_scn[0] == model_strct and sn in strct_scn[1]]
# #             sn_strct_acc = np.mean(sn_strct_accuracies)
# #             strct_sns_acc_dict[model_strct].append(sn_strct_acc)
# #     ind = np.arange(len(scns))
# #     for i, model_strct_scn_accs in enumerate(strct_sns_acc_dict.items()):
# #         model_strct, scn_accs = model_strct_scn_accs
# #         plt.bar(ind + i * width, scn_accs, width, label=model_strct)
# #         for j in range(len(scn_accs)):
# #             plt.text(ind[j] + i * width, scn_accs[j] + 0.05, str(round(scn_accs[j], 3)), horizontalalignment='center',
# #                      verticalalignment='center')
# #     plt.ylim(0.0, 1.1)
# #     plt.ylabel('Average Accuracy')
# #     plt.title('Accuracy by model structure and scenario')
# #     plt.xticks(ind + width / 2, scns)
# #     plt.legend(loc=4)
# #     plt.show()
# #
# # plot scenarios-based training history ##################################################################
# plt.rcParams["figure.figsize"] = (15, 15)
# SMALL_SIZE = 18
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 22
#
# plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)
#
# fig, ax = plt.subplots(nrows=len(locs), ncols=len(freqs))
# for i, lc in enumerate(rename_map.values()):
#     for j, fq in enumerate(rename_freq_map.values()):
#         strct_sn_hists = [(strct_scn[0], hist_eval[0].history['val_accuracy']) for strct_scn, hist_eval in
#                           scenario_train_histories_rename.items() if (lc, fq) == strct_scn[1]]
#         for strct, sn_hist in strct_sn_hists:
#             ax[i][j].plot(sn_hist, label=strct)
#         ax[i][j].set_xlabel('Epoch')
#         ax[i][j].set_ylabel('Validation accuracy')
#         # ax[i][j].set_ylim(-0.1, 3.0)
#         ax[i][j].set_title(str(lc) + ', ' + str(fq))
#         # ax[i][j].legend(loc='best')
# plt.tight_layout()
# plt.show()
#
# # visualize intermidiate CNN kernels ############################
# # first sample ###############
# scn = ('han', '24g')
# clss = 'sta'
# np.random.seed(3)
# dataset = scenario_data[scn]
# sample = dataset['x'][np.random.choice(np.where(dataset['y'] == clss)[0])]
# sample_normalized = sc.transform(sample)
# sample_normalized_batch = np.expand_dims(sample_normalized, axis=0)
#
# plt.rcParams["figure.figsize"] = (8, 4)
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.plot(sample[:, 0], color=color)
# ax1.set_xlabel('Timesteps')
# ax1.set_ylabel('Phase', color=color)
#
#
# color = 'tab:blue'
# ax2 = ax1.twinx()
# ax2.plot(sample[:, 1], color=color)
# ax2.set_xlabel('Timesteps')
# ax2.set_ylabel('RSS', color=color)
#
# fig.tight_layout()
# plt.title('Scenario: ' + rename_map[scn[0]] + ',' + rename_freq_map[scn[1]] + '; ' + clss)
# plt.show()
#
# # get corresponding model
# model = scenario_train_histories[('CNN', scn)][0].model
#
# # xy = scenario_data[scn]
# # x = np.array([sc.transform(x_raw) for x_raw in xy['x']])
# # y = encoder.transform(xy['y'].reshape(-1, 1)).toarray()
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, shuffle=True)
# # eval = model.evaluate(x=x_test, y=y_test)
#
# classes = model.predict(sample_normalized_batch)
# print("Predicted class is:", encoder.inverse_transform(classes))
#
# # Creates a model that will return these outputs, given the model input
# layer_outputs = [layer.output for layer in model.layers[:9] if 'batch_normalization' not in layer.name]
# # Extracts the outputs of the top 12 layers
# from tensorflow.python.keras import models
#
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(sample_normalized_batch)
#
# layer_names = []
# for layer in model.layers[:9]:
#     if 'batch_normalization' not in layer.name:
#         layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot
#
# first_layer_activation = activations[0]
# print(first_layer_activation.shape)
# plt.plot(first_layer_activation[0, :, 0])
# plt.show()
#
# images_per_row = 8
# dpi = 180
# sub_fig_size = 5
# for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
#     n_features = layer_activation.shape[-1] # Number of features in the feature map
#     size = dpi * sub_fig_size
#
#     n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
#     display_grid = np.zeros((size * n_cols, images_per_row * size, 3), dtype=int)
#     for col in range(n_cols): # Tiles each filter into a big horizontal grid
#         for row in range(images_per_row):
#             channel_image = layer_activation[0, :, col * images_per_row + row]
#             fig = plt.figure(figsize=(sub_fig_size, sub_fig_size))
#             ax = fig.add_subplot(111)
#             plt.plot(channel_image, linewidth = 3)
#             plt.axis('off')
#             channel_image = get_img_from_fig(fig, dpi=dpi)
#             display_grid[col * size : (col + 1) * size, # Displays the grid
#                          row * size : (row + 1) * size, :] = channel_image
#             plt.clf()
#     scale = 1. / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#     plt.show()
# # visualize Attention weights  ############################
#
# # plot ROC curve for model  ############################
classes = ['Distractor','Target', 'Novelty']
model = history_rnn.model
model_name = 'RNN'

plt.rcParams["figure.figsize"] = (15, 15)
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
BIGGER_BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

print('Plotting scenario: ' + str(scn))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3, shuffle=True)

eval = model.evaluate(x=x_test, y=y_test)
y_score = model.predict(x_test)

plot_roc_multiclass(n_classes=3, y_score=y_score, y_test=y_test, classes=classes, zoom=False)
plt.show()

# # plot CM  for model  ###############################

#Get the confusion matrix
cf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_score, axis=1))
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')
plt.rc('font', size=BIGGER_BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_BIGGER_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_BIGGER_SIZE)

ax.set_title('Epoch Confusion Matrix for {0}'.format(history_rnn));
ax.set_xlabel('\nPredicted Epoch Label')
ax.set_ylabel('Actual Epoch Label ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)

## Display the visualization of the Confusion Matrix.
plt.show()