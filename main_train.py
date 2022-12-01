import pickle

from learning.models import EEGCNNNet
from learning.train import score_model

# analysis parameters ######################################################################################


print("Loading RDF")
rdf = pickle.load(open('rdf.p', 'rb'))
#
# # discriminant test  ####################################################################################################
# event_names = ["Distractor", "Target"]
# event_filters = [lambda x: x.dtn_onffset and x.dtn==dtnn_types["Distractor"],
#                  lambda x: x.dtn_onffset and x.dtn==dtnn_types["Target"]]
# eeg_epochs, eeg_event_ids = rdf.get_eeg_epochs(event_names, event_filters)
#
# x, y = epochs_to_class_samples(eeg_epochs, eeg_event_ids)
# epoch_shape = x.shape[1:]
# x = np.reshape(x, newshape=(len(x), -1))
# sm = SMOTE(random_state=42)
# x, y = sm.fit_resample(x, y)
# x = np.reshape(x, newshape=(len(x), ) + epoch_shape)  # reshape back x after resampling
#
# # z-norm along channels
# x = (x - np.mean(x, axis=(0, 2), keepdims=True)) / np.std(x, axis=(0, 2), keepdims=True)
#
# # sanity check the channels
# x_distractors = x[:, eeg_montage.ch_names.index('CPz'), :][y==0]
# x_targets = x[:, eeg_montage.ch_names.index('CPz'), :][y==1]
# x_distractors = np.mean(x_distractors, axis=0)
# x_targets = np.mean(x_targets, axis=0)
# plt.plot(x_distractors)
# plt.plot(x_targets)
# plt.show()
#
# epochs_to_class_samples(rdf, event_names, event_filters, rebalance=True)

x = pickle.load(open('x.p', 'rb'))
y = pickle.load(open('y.p', 'rb'))
model = EEGCNNNet(in_shape=x.shape, num_classes=2)
model = score_model(x, y, model)
