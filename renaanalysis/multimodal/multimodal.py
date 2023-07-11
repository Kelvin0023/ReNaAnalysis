from typing import List

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from renaanalysis.learning.MutiInputDataset import MultiInputDataset
from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica, rebalance_classes


class PhysioArray:
    """
    when rebalance is applied, is this array rebalanced by channel or not
    @atttribute array: the raw array as was initially given at class's instantiation
    @atttribute array_preprocessed: the array after preprocessing, e.g. znorm, pca, ica
    @attribute sampling_rate
    @attribute physio_type
    @attribute is_rebalance_by_channel: when supersampling method such as SMOTE is applied as part of rebalancing,
    whether to apply it by channel or not
    """
    def __init__(self, array: np.ndarray, sampling_rate: float, physio_type: str, is_rebalance_by_channel=False, dataset_name=''):
        self.array = array
        self.sampling_rate = sampling_rate
        self.physio_type = physio_type
        self.is_rebalance_by_channel = is_rebalance_by_channel
        self.data_processor = dict()
        self.dataset_name = dataset_name

        self.array_preprocessed = None

    def __getitem__(self, item):
        return self.array[item]

    def __len__(self):
        return len(self.array)

    def __str__(self):
        data_preprocessor_str = '-'.join(self.data_processor.keys())
        return f'{self.dataset_name}_{self.physio_type}_{data_preprocessor_str}'

    def apply_znorm_by_trial(self):
        self.array_preprocessed = z_norm_by_trial(self.array)
        self.data_processor['znorm'] = True

    def apply_pca_ica(self, n_top_components=20):
        """
        @param n_top_components: number of top components to keep for pca and ica
        @return:
        """
        if 'pca_ica_components' in self.data_processor.keys() and n_top_components == self.data_processor['pca_ica_components']:
            return
        if 'znorm' not in self.data_processor.keys():
            self.apply_znorm_by_trial()
        self.array_preprocessed, self.data_processor['pca'], self.data_processor['ica'] = compute_pca_ica(self.array_preprocessed, n_top_components)
        self.data_processor['pca_ica_components'] = n_top_components

    def get_pca_ica_array(self):
        if 'pca_ica_components' not in self.data_processor.keys():
            raise ValueError('pca ica has not been applied to this array')
        return self.array_preprocessed


class MultiModalArrays:
    """

    """
    def __init__(self, physio_arrays: List[PhysioArray], labels_array: np.ndarray =None, dataset_name: str='', event_viz_colors: dict=None, rebalance_method='SMOTE'):
        """

        @param physio_arrays:
        @param labels_array:
        @param dataset_name:
        @param event_viz_colors:
        @param rebalance_method: can be SMOTE or class_weight
        """
        assert len(physio_arrays) > 0, 'physio_arrays must be a non-empty list of PhysioArray'
        assert np.all(physio_arrays[0].array.shape[0] == [parray.array.shape[0] for parray in physio_arrays]), 'all physio arrays must have the same number of trials/epochs'

        self.physio_arrays = physio_arrays
        self.labels_array = labels_array
        self.dataset_name = dataset_name
        self.event_viz_colors = event_viz_colors
        self._physio_types = [parray.physio_type for parray in physio_arrays]
        self._physio_types_arrays = dict(zip(self._physio_types, self.physio_arrays))
        self.rebalance_method = rebalance_method

        self.event_names = list(event_viz_colors.keys()) if event_viz_colors is not None else None

        self.test_indices = None
        self.train_indices = None
        self.training_val_split_indices = None

        self.labels_encoder = None
        self.label_onehot_encoder = None
        self._encoder = None
        if self.labels_array is not None:
            self.label_encoder = LabelEncoder()
            self.labels_array = self.label_encoder.fit_transform(self.labels_array)

            self.label_onehot_encoder = preprocessing.OneHotEncoder()
            self.label_onehot_encoder.fit(self.labels_array.reshape(-1, 1))

    def keys(self):
        return self._physio_types_arrays.keys()

    def __getitem__(self, physio_type):
        return self._physio_types_arrays[physio_type]

    def __len__(self):
        return len(self.physio_arrays)

    def __iter__(self):
        return iter(self.physio_arrays)

    def __str__(self):
        return '|'.join([str(parray) for parray in self.physio_arrays])

    def get_rebalanced_dataloader_fold(self, fold_index, batch_size, random_seed=None, device=None):
        """
        get the dataloader for a given fold
        this function must be called after training_val_split
        @param fold:
        @param random_seed:
        @return:
        """
        assert self.labels_array is not None, 'labels array must be provided to use rebalancing'
        assert self._encoder is not None, 'get_label_encoder_criterion_for_model must be called before get_rebalanced_dataloader_fold'
        training_indices, val_indices = self.training_val_split_indices[fold_index]
        x_train = []
        x_val = []
        y_train = self.labels_array[training_indices]
        y_val = self.labels_array[val_indices]
        rebalanced_labels = []
        for parray in self.physio_arrays:
            this_x_train, this_y_train = parray[training_indices], y_train
            if self.rebalance_method == 'SMOTE':
                this_x_train, this_y_train = rebalance_classes(parray[training_indices], y_train, by_channel=parray.is_rebalance_by_channel, random_seed=random_seed)
            x_train.append(torch.Tensor(this_x_train).to(device))
            rebalanced_labels.append(this_y_train)
            x_val.append(torch.Tensor(parray[val_indices]).to(device))
        assert np.all([label_set == rebalanced_labels[0] for label_set in rebalanced_labels])
        y_train = rebalanced_labels[0]

        y_train_encoded = self._encoder(y_train)
        y_val_encoded = self._encoder(y_val)

        y_train_encoded = torch.Tensor(y_train_encoded)
        y_val_encoded = torch.Tensor(y_val_encoded)

        if len(x_train) == 1:
            dataset_class = TensorDataset
        else:
            dataset_class = MultiInputDataset
        train_dataset = dataset_class(x_train, y_train_encoded)
        val_dataset = dataset_class(x_val, y_val_encoded)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

    def training_val_split(self, n_folds, val_size=0.1, random_seed=None):
        """
        split the train set into training and validation sets

        this function must be called after train_test_split
        @return:
        """
        assert self.train_indices is not None, 'train indices have not been set, please call train_test_split first'
        if self.labels_array is not None:
            skf = StratifiedShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
            for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array, self.labels_array)):
                self.training_val_split_indices.append((train, val))
        else:
            skf = ShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
            for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array)):
                self.training_val_split_indices.append((train, val))

    def train_test_split(self, test_size=0.1, random_seed=None):
        """
        split the raw dataset in to train and test sets
        if label is provided,
            uses Stratified ShuffleSplit from sklearn, ensures that the train and test sets have the same ratio between
            each class.
        else:
            uses simple train-test split from sklearn
        The array must have labels for this to work.
        @return:
        """
        if self.labels_array is not None:
            skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
            self.train_indices, self.test_indices = [(train, test) for train, test in skf.split(self.physio_arrays[0].array, self.labels_array)][0]
            assert np.all(np.unique(self.labels_array[self.test_indices]) == np.unique(self.labels_array[self.train_indices])), "train and test labels are not the same"
            assert len(np.unique(self.labels_array[self.test_indices])) == len(self.event_names), "number of unique labels is not the same as number of event names"
        else:
            self.train_indices, self.test_indices = train_test_split(list(range(self.physio_arrays[0].array.shape[0])), test_size=test_size, random_state=random_seed)


    def get_random_sample(self, preprocessed=False, convert_to_tensor=False, device=None):
        """
        @return: a random sample from the array
        """
        random_sample_index = np.random.randint(0, len(self.physio_arrays[0]))
        rtn = [(parray.array[random_sample_index] if not preprocessed else parray.array_preprocessed[random_sample_index]) for parray in self.physio_arrays]
        if convert_to_tensor:
            rtn = [torch.tensor(r, dtype=torch.float32, device=device) for r in rtn]
        return rtn if len(rtn) > 1 else rtn[0]

    def get_class_weight(self, convert_to_tensor=False, device=None):
        assert self.labels_array is not None, "Class weight needs labels array but labels is not provided"
        unique_classes, counts = np.unique(self.labels_array, return_counts=True)
        class_proportions = counts / len(self.labels_array)
        class_weights = 1/class_proportions
        if convert_to_tensor:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return class_weights

    def encode_labels(self):
        pass  # TODO

    def get_label_encoder_criterion_for_model(self, model, device=None):
        """
        this function must be called

        @param model:
        @param device:
        @param class_weights:
        @return:
        """
        rand_input = self.get_random_sample(convert_to_tensor=True, device=device)
        with torch.no_grad():
            model.eval()
            output_shape = model.to(device)(rand_input).shape[1]

        if output_shape == 1:
            assert len(np.unique(self.labels_array)) == 2, "Model only has one output node. But given Y has more than two classes. Binary classification model should have 2 classes"
            self._encoder = lambda y: self.label_encoder.transform(y).reshape(-1, 1)
            # _decoder = lambda y: label_encoder.inverse_transform(y.reshape(-1, 1))
            criterion = nn.BCELoss(reduction='mean')
            last_activation = nn.Sigmoid()
        else:

            self._encoder = lambda y: self.label_onehot_encoder.transform(y.reshape(-1, 1)).toarray()
            # _decoder = lambda y: label_encoder.inverse_transform(y.reshape(-1, 1))
            criterion = nn.CrossEntropyLoss(weight=self.get_class_weight(True, device) if self.rebalance_method=='class_weight' else None)
            last_activation = nn.Softmax(dim=1)

        return criterion, last_activation

    def get_encoder_function(self):
        return self._encoder