import os
import pickle

import numpy as np

dataloader_root = r'C:\Data\DataLoaders\auditory_oddball'  # IMPORTANT: change this to your data root

meta_info_path = os.path.join(dataloader_root, 'meta_info.p')
test_loader_path = os.path.join(dataloader_root, 'test_dataloader.p')

meta_info = pickle.load(open(meta_info_path, 'rb'))

# print the meta info
print(f'Loaded dataset with meta info: {meta_info}')

test_loader = pickle.load(open(test_loader_path, 'rb'))
print(f'Number of test samples: {len(test_loader.dataset)}')
print(f'Number of test batches: {len(test_loader)}')

for fold_index in range(meta_info['n_folds']):
    train_loader_path = os.path.join(dataloader_root, f'train_dataloader_fold{fold_index}.p')
    val_loader_path = os.path.join(dataloader_root, f'val_dataloader_fold{fold_index}.p')

    train_loader = pickle.load(open(train_loader_path, 'rb'))
    val_loader = pickle.load(open(val_loader_path, 'rb'))

    print(f'Fold {fold_index}')
    print(f'Number of training samples: {len(train_loader.dataset)} in {len(train_loader)} batches, with labels {np.unique(train_loader.dataset.labels.numpy(), return_counts=True)}')
    print(f'Number of validation samples: {len(val_loader.dataset)} in {len(val_loader)} batches, with labels {np.unique(val_loader.dataset.labels.numpy(), return_counts=True)}')
    print('\n')

    # using the following code to iterate through the training data
    # for epoch in range(epochs):
    #     for batch_data in train_loader:
    #         y = batch_data['y']  # this is the label
    #         x = batch_data  # this is the input data, it is a dictionary including different physiological modalities (e.g., eeg), and other information (e.g., channel location, subject, session, etc.)
    #         # put your training code here
