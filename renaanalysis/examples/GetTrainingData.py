import os

import torch

from renaanalysis.utils.dataset_utils import get_dataset

'''
use 
    auditory_oddball for erp
    BCICIVA for motor imagery
    DEAP for emotion detection
'''
dataset_name = 'auditory_oddball'
data_root = r'D:\Dropbox\Dropbox\EEGDatasets\auditory_oddball_openneuro'  # IMPORTANT: change this to your data root
export_data_root = 'C:/Data'  # location to save preloaded data
reject = None  # whether to apply auto rejection, can be set to None, 'auto'. Note that 'auto' will take a long time to run
is_pca_ica = False # apply pca and ica on data as a preprocessing step
is_regenerate_epochs = True  # whether to regenerate epochs, if false, will load from export_data_root
random_seed = 0  # just a random seed
eeg_baseline = None  # not applying any baseline correction


# training parameters
n_folds = 2
epochs = 100
test_size = 0.1
val_size = 0.1
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
mmarray: MultiModalArray is our data structure that contains all the data
it also handles the splitting of data into training, validation and test sets
'''
mmarray_fn = f'{dataset_name}_mmarray.p'
mmarray_path = os.path.join(export_data_root, mmarray_fn)
mmarray = get_dataset(dataset_name, epochs_root=export_data_root, dataset_root=data_root,
                      reject=reject, is_apply_pca_ica_eeg=is_pca_ica, is_regenerate_epochs=is_regenerate_epochs,
                      random_seed=random_seed, rebalance_method="class_weight", filename=mmarray_path,
                      eeg_baseline=eeg_baseline)

mmarray.test_train_val_split(n_folds, test_size=test_size, val_size=val_size, random_seed=random_seed)  # split data into training, validation and test sets
test_dataloader = mmarray.get_test_dataloader(batch_size=batch_size, device=device) # get test dataloader, this is outside of the fold loop because we only need one test set

for fold_index in range(n_folds):
    train_dataloader, val_dataloader = mmarray.get_dataloader_fold(fold_index, batch_size=batch_size, is_rebalance_training=True, random_seed=random_seed, device=device, shuffle_within_batches=False)
    for epoch in range(epochs):
        for batch_data in train_dataloader:
            y = batch_data['y']  # this is the label
            x = batch_data  # this is the input data, it is a dictionary including different physiological modalities (e.g., eeg), and other information (e.g., channel location, subject, session, etc.)
            # put your training code here
