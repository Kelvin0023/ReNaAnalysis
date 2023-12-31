import copy
import itertools
import math
import pickle
import warnings
from typing import List

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

from renaanalysis.learning.HT import HierarchicalTransformerContrastivePretrain
from renaanalysis.learning.data_utils import fill_batch_with_none_indices
from renaanalysis.multimodal.BatchIterator import OrderedBatchIterator
from renaanalysis.multimodal.MultimodalDataset import MultiModalDataset
from renaanalysis.multimodal.PhysioArray import PhysioArray
from renaanalysis.utils.TorchUtils import batch_to_tensor
from renaanalysis.multimodal.MultimodalDataset import MultiModalDataset
from renaanalysis.multimodal.PhysioArray import PhysioArray
from renaanalysis.utils.TorchUtils import batch_to_tensor
from renaanalysis.params.params import TaskName
from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica, rebalance_classes



class PhysioArray:
    """
    when rebalance is applied, is this array rebalanced by channel or not
    @attribute array: the raw array as was initially given at class's instantiation
    @attribute meta_info:

    @atttribute array: the raw array as was initially given at class's instantiation
    @atttribute array_preprocessed: the array after preprocessing, e.g. znorm, pca, ica
    @attribute sampling_rate
    @attribute physio_type
    @attribute is_rebalance_by_channel: when supersampling method such as SMOTE is applied as part of rebalancing,
    whether to apply it by channel or not
    """
    def __init__(self, array: np.ndarray, meta_info: dict, sampling_rate: float, physio_type: str, is_rebalance_by_channel=False, dataset_name='', ch_names=None):
        assert np.all(array.shape[0] == np.array([len(m) for m in meta_info.values()])), 'all metainfo in a physio array must have the same number of trials/epochs'
        self.array = array
        self.meta_info = meta_info

        self.meta_info_encoders = dict()
        self.meta_info_encoded = dict()
        self.encode_meta_info()

        self.sampling_rate = sampling_rate
        self.physio_type = physio_type
        self.is_rebalance_by_channel = is_rebalance_by_channel
        self.data_processor = dict()
        self.dataset_name = dataset_name
        self.ch_names = ch_names

        self.array_preprocessed = None

    def __getitem__(self, item):
        if self.data_processor is not None:
            return self.array_preprocessed[item]
        else:
            return self.array[item]

    def __len__(self):
        return len(self.array)

    def __str__(self):
        data_preprocessor_str = '-'.join(self.data_processor.keys())
        return f'{self.dataset_name}_{self.physio_type}_{data_preprocessor_str}'

    def encode_meta_info(self):
        """
        will encode any meta info that is not numeric
        @return:
        """
        for name, value in self.meta_info.items():
            if value.dtype == object:
                self.meta_info_encoders[name] = LabelEncoder()
                self.meta_info_encoded[name] = self.meta_info_encoders[name].fit_transform(value)
            else:
                self.meta_info_encoded[name] = value

    def concatenate(self, other_physio_array):
        """
        concatenate two physio arrays, assuming they have the same meta info
        @param other_physio_array:
        @return:
        """
        assert np.all(self.meta_info.keys() == other_physio_array.meta_info.keys()), 'both arrays must have the same meta info keys'
        assert self.physio_type == other_physio_array.physio_type, 'both arrays must have the same physio type'
        assert self.array.shape[1:-1] == other_physio_array.array.shape[1:-1], 'both arrays must have the same number of channels and time stamps'
        if self.array_preprocessed is not None and other_physio_array.array_preprocessed is not None:
            assert self.array_preprocessed.shape[1:-1] == other_physio_array.array_preprocessed.shape[1:-1], 'both arrays must have the same number of channels and time stamps'
            self.array_preprocessed = np.concatenate([self.array_preprocessed, other_physio_array.array_preprocessed])
        else:
            assert self.array_preprocessed is None and other_physio_array.array_preprocessed is None, 'both preprocessed arrays must have the same number of channels and time stamps'
        assert self.data_processor.keys() == other_physio_array.data_processor.keys(), 'both arrays must have the same data processor'
        # assert self.data_processor.values() == other_physio_array.data_processor.values(), 'both arrays must have the same data processor'
        self.array = np.concatenate([self.array, other_physio_array.array])
        for name, value in other_physio_array.meta_info.items():
            self.meta_info[name] = np.concatenate([self.meta_info[name], value])
        if self.meta_info_encoded is not None and other_physio_array.meta_info_encoded is not None:
            for name, value in other_physio_array.meta_info_encoded.items():
                self.meta_info_encoded[name] = np.concatenate([self.meta_info_encoded[name], value])
        return self

    def get_meta_info_by_name(self, meta_info_name):
        return self.meta_info[meta_info_name]

    def get_meta_info(self, index, encoded=False):
        return {k: v[index] for k, v in (self.meta_info_encoded if encoded else self.meta_info).items()}

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
    def __init__(self, physio_arrays: List[PhysioArray], labels_array: np.ndarray =None, dataset_name: str= '', event_viz_colors: dict=None, rebalance_method='SMOTE',
                 filename=None, experiment_info: dict=None):
        """
        mmarray will assume the ordered set of labels correpond to each event in event_viz_colors
        @param physio_arrays:
        @param labels_array:
        @param dataset_name:
        @param event_viz_colors:
        @param rebalance_method: can be SMOTE or class_weight
        """
        self._encoder_object = None
        assert len(physio_arrays) > 0, 'physio_arrays must be a non-empty list of PhysioArray'
        assert np.all(len(physio_arrays[0]) == np.array([len(parray) for parray in physio_arrays])), 'all physio arrays must have the same number of trials/epochs'

        self.physio_arrays = physio_arrays
        self.labels_array = labels_array
        self.dataset_name = dataset_name

        self.event_viz_colors = event_viz_colors

        self._physio_types = [parray.physio_type for parray in physio_arrays]
        self._physio_types_arrays = dict(zip(self._physio_types, self.physio_arrays))
        self.rebalance_method = rebalance_method
        self.experiment_info = experiment_info

        self.event_names = list(event_viz_colors.keys()) if event_viz_colors is not None else None

        if self.labels_array is not None:
            self.event_ids = {label: e_name for label, e_name in zip(self.event_names, set(self.labels_array))}
            self.event_id_viz_colors = {self.event_ids[label]: color for label, color in self.event_viz_colors.items()}
        self.test_indices = None
        self.train_indices = None
        self.training_val_split_indices = None

        self.labels_encoder = None
        self.label_onehot_encoder = None
        self._encoder = None
        if self.labels_array is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels_array)

            self.label_onehot_encoder = preprocessing.OneHotEncoder()
            self.label_onehot_encoder.fit(self.labels_array.reshape(-1, 1))

        # for ordered batches
        self.test_batch_sample_indices = None
        self.val_batch_sample_indices = None
        self.train_batch_sample_indices = None
        self.filename = filename

        self.multi_modal_dataset = MultiModalDataset(self.physio_arrays, self.labels_array)
        # print('check if physio and label arrays are the same after previous call')

    def keys(self):
        return self._physio_types_arrays.keys()

    def __getitem__(self, physio_type):
        return self._physio_types_arrays[physio_type]

    def __len__(self):
        return len(self.physio_arrays)

    def __iter__(self):
        return iter(self.physio_arrays)

    def __str__(self):
        return '-'.join([str(parray) for parray in self.physio_arrays])

    def get_num_samples(self):
        return len(self.physio_arrays[0])

    def get_indices_by_subject_run(self, subject, run):
        return [i for i, (s, r) in enumerate(zip(self.experiment_info['subject_id'], self.experiment_info['run'])) if s == subject and r == run]

    def get_indices_from_picks(self, picks):
        train_indices_fold = []
        val_indices_fold = []
        for n_fold, subject_dict in enumerate(picks['subjects']):
            this_train_indices = []
            this_val_indices = []
            for train_subject in subject_dict['train']:
                for train_run in picks['run'][n_fold]['train']:
                    this_train_indices += self.get_indices_by_subject_run(train_subject, train_run)
            for val_subject in subject_dict['val']:
                for val_run in picks['run'][n_fold]['val']:
                    this_val_indices += self.get_indices_by_subject_run(val_subject, val_run)
            train_indices_fold.append(this_train_indices)
            val_indices_fold.append(this_val_indices)
        return [(np.array(i), np.array(j)) for (i, j) in zip(train_indices_fold, val_indices_fold)]

    def get_dataloader_fold(self, fold_index, batch_size, random_seed=None, device=None, encode_y=True, *args, **kwargs):
        """
        get the dataloader for a given fold
        this function must be called after training_val_split
        @param fold:
        @param random_seed:
        @param picks: dict of subject and run to pick for train and val respectively, if None, dataloader split it randomly
        @return:
        """
        if self.rebalance_method == 'class_weight' and self._encoder_object is not None and isinstance(self._encoder_object, LabelEncoder):
            warnings.warn("Using class_weight as rebalancing method while encoder is LabelEncoder because BCELoss can not apply class weights ")

        training_indices, val_indices = self.training_val_split_indices[fold_index]
        labels = self.get_encoded_labels() if encode_y else np.copy(self.labels_array)

        val_dataset = MultiModalDataset(self.physio_arrays, labels=labels, indices=val_indices)

        # rebalance training set
        if self.rebalance_method == 'SMOTE':
            if not encode_y: warnings.warn("Using smote may not work when not encoding y, please double check")
            train_dataset = MultiModalDataset(self.physio_arrays, labels=self.labels_array, indices=training_indices)
            train_dataset.get_rebalanced_set(random_seed=random_seed, encoder=self._encoder)
        else:
            train_dataset = MultiModalDataset(self.physio_arrays, labels=labels, indices=training_indices)
            # print('check if physio and label arrays are the same after previous call')

        val_dataset.to_tensor(device=device)
        train_dataset.to_tensor(device=device)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader


    def _training_val_split(self, n_folds, val_size=0.1, random_seed=None, split_picks=None):
        """
        split the train set into training and validation sets

        this function must be called after train_test_split

        @param val_size: proportion of to the dataset excluding the test set
        @return:
        """
        assert self.train_indices is not None, 'train indices have not been set, please call train_test_split first'
        if split_picks is None:
            self.training_val_split_indices = []
            if self.labels_array is not None:
                skf = StratifiedShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
                for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array[self.train_indices], self.labels_array[self.train_indices])):
                    self.training_val_split_indices.append((self.train_indices[train], self.train_indices[val]))
            else:
                skf = ShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
                for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array[self.train_indices])):
                    self.training_val_split_indices.append((np.array(self.train_indices)[train], np.array(self.train_indices)[val]))
        else:
            print("training_val_split: Using predefined picks, n_fold and val_size will be ignored")
            self.training_val_split_indices = self.get_indices_from_picks(split_picks)
        self.save()


    def _train_test_split(self, test_size=0.1, random_seed=None):
        """
        split the raw dataset in to train and test sets
        if label is provided,
            uses Stratified ShuffleSplit from sklearn, ensures that the train and test sets have the same ratio between
            each class.
        else:
            uses simple train-test split from sklearn
        The array must have labels for this to work.

        @param test_size: proportion of the ENTIRE dataset
        @return:
        """
        if self.labels_array is not None:
            if test_size != 0:
                skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
                self.train_indices, self.test_indices = [(train, test) for train, test in skf.split(self.physio_arrays[0].array, self.labels_array)][0]
                assert np.all(np.unique(self.labels_array[self.test_indices]) == np.unique(self.labels_array[self.train_indices])), "train and test labels are not the same"
                assert len(np.unique(self.labels_array[self.test_indices])) == len(self.event_names), "number of unique labels is not the same as number of event names"
            else:
                self.train_indices = np.arange(self.physio_arrays[0].array.shape[0])
                self.test_indices = np.array([], dtype=int)
        else:
            self.train_indices, self.test_indices = train_test_split(list(range(self.physio_arrays[0].array.shape[0])), test_size=test_size, random_state=random_seed, stratify=self.labels_array)
        self.save()


    def test_train_val_split(self, n_folds, test_size=0.1, val_size=0.1, random_seed=None, split_picks=None):
        """
        create indices for test, train and validation sets

        @param n_folds:
        @param test_size: proportional to the size of the ENTIRE dataset
        @param val_size: proportional to the size of the ENTIRE dataset
        @param random_seed:
        @return:
        """
        self._train_test_split(test_size=test_size, random_seed=random_seed)
        val_size = val_size / (1 - test_size)  # adjust the val size to be a percentage of the training set
        if split_picks is not None:
            n_folds = len(split_picks['subjects'])
        self._training_val_split(n_folds=n_folds, val_size=val_size, random_seed=random_seed, split_picks=split_picks)
        for i in range(n_folds):
            assert set(self.training_val_split_indices[i][0]).intersection(set(self.test_indices)) == set(), 'train and test sets are not disjoint'
            assert set(self.training_val_split_indices[i][0]).intersection(set(self.training_val_split_indices[i][1])) == set(), 'train and val sets are not disjoint'
            assert set(self.training_val_split_indices[i][1]).intersection(set(self.test_indices)) == set(), 'val and test sets are not disjoint'


    def set_training_val_set(self, train_indices, val_indices):
        """
        set the train indices
        @param train_indices:
        @return:
        """
        self.training_val_split_indices = []
        if isinstance(train_indices, list) and isinstance(val_indices, list):
            assert len(train_indices) == len(val_indices), 'train and val must have the same number of folds'
            for i in range(len(train_indices)):
                self.training_val_split_indices.append((np.array(train_indices[i]) if isinstance(train_indices[i], list) else train_indices[i], np.array(val_indices[i]) if isinstance(val_indices[i], list) else val_indices[i]))
        else:
            self.training_val_split_indices.append((np.array(train_indices), np.array(val_indices)))
        self.save()


    def set_train_indices(self, train_indices):
        """
        set the train indices
        @param train_indices:
        @return:
        """
        self.train_indices = train_indices
        self.save()

    def set_test_indices(self, test_indices):
        """
        set the test indices
        @param test_indices:
        @return:
        """
        self.test_indices = test_indices
        self.save()

    def get_ordered_test_indices(self):
        assert self.test_batch_sample_indices is not None, 'test batch sample indices have not been set, please call training_val_test_split_ordered_by_subject_run'
        indices = self.test_batch_sample_indices.reshape(-1)
        indices = indices[indices != None].astype(int)
        return indices

    def get_test_set(self, use_ordered, convert_to_tensor=False, device=None):
        """
        get the test set
        @return:
        """
        if use_ordered:
            test_indices = self.get_ordered_test_indices()
            if self.labels_array is None:
                warnings.warn('labels array is None, make sure label is not needed for this model')
            test_set = self.multi_modal_dataset[test_indices]
        else:
            assert self.test_indices is not None or self.test_batch_sample_indices is not None, 'test indices have not been set, please call train_test_split first, or training_val_test_split_ordered_by_subject_run'
            test_indices = self.test_indices
            if self.labels_array is None:
                warnings.warn('labels array is None, make sure label is not needed for this model')
            test_set = MultiModalDataset(self.physio_arrays, labels=self.get_encoded_labels(), indices=test_indices)

        if convert_to_tensor:
            test_set.to_tensor(device=device)
        return test_set


    def get_random_sample(self, convert_to_tensor=False, device=None):
        """
        @return: a random sample from each of the physio arrays
        """
        random_sample_index = np.random.randint(0, len(self.physio_arrays[0]))
        sample = self.multi_modal_dataset[random_sample_index:random_sample_index+1]
        if convert_to_tensor:
            sample = batch_to_tensor(sample, device=device)
        return sample

    def get_class_weight(self, convert_to_tensor=False, device=None):
        """
        An example of one-hot encoded label array, the original labels are [0, 6]
        The corresponding cw is:
                     Count
        0 -> [1, 0]  100
        6 -> [0, 1]  200
        cw:  [3, 1.5]
        because pytorch treat [1, 0] as the first class and [0, 1] as the second class. However, the
        count for unique one-hot encoded label came out of np.unique is in the reverse order [0, 1] and [1, 0].
        the count needs to be reversed accordingly.

        TODO check when adding new classes
        @param convert_to_tensor:
        @param device:
        @return:
        """
        assert self.labels_array is not None, "Class weight needs labels array but labels is not provided"
        encoded_labels = self._encoder(self.labels_array)
        if len(encoded_labels.shape) == 2:
            unique_classes, counts = np.unique(encoded_labels, return_counts=True, axis=0)
            counts = counts[::-1]  # refer to docstring
        elif len(encoded_labels.shape) == 1:
            unique_classes, counts = np.unique(encoded_labels, return_counts=True)
        else:
            raise ValueError("encoded labels should be either 1d or 2d array")
        # labels_as_strings = np.array([str(label) for label in unique_classes])
        # counts = counts[np.flip(np.argsort(labels_as_strings))]
        class_proportions = counts / len(self.labels_array)
        class_weights = 1/class_proportions
        if convert_to_tensor:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return class_weights # reverse the class weights because

    def encode_labels(self):
        pass  # TODO

    def create_label_encoder(self, output_shape):
        """
        create a label encoder for the dataset
        @return:
        """
        if output_shape == 1:
            assert len(np.unique(
                self.labels_array)) == 2, "Model only has one output node. But given Y has more than two classes. Binary classification model should have 2 classes"
            # self._encoder = lambda y: self.label_onehot_encoder.transform(y.reshape(-1, 1)).toarray()
            self._encoder_object = self.label_encoder
            self._encoder = lambda y: self.label_encoder.transform(y.reshape(-1, 1))
        else:
            self._encoder_object = self.label_onehot_encoder
            self._encoder = lambda y: self.label_onehot_encoder.transform(y.reshape(-1, 1)).toarray()


    def get_label_encoder_criterion_for_model(self, model, device=None):
        """
        this function must be called

        @param model:
        @param device:
        @param class_weights:
        @return:
        """
        assert not isinstance(model, HierarchicalTransformerContrastivePretrain), "Model is HierarchicalTransformerContrastivePretrain, which does not have label encoder, please use function get_critierion_for_pretrain"
        rand_input = self.get_random_sample(convert_to_tensor=True, device=device)
        with torch.no_grad():
            model.eval()
            rand_input= rand_input if isinstance(rand_input, tuple) else (rand_input,)
            output_shape = model.to(device)(*rand_input).shape[1]
            # output_shape = model.to(device)(rand_input['eeg'][:, None, :, :]).shape[1]

        self.create_label_encoder(output_shape)
        if output_shape == 1:
            criterion = nn.BCELoss(reduction='mean')
            last_activation = nn.Sigmoid()
        else:
            criterion = nn.CrossEntropyLoss(weight=self.get_class_weight(True, device) if self.rebalance_method == 'class_weight' else None)
            last_activation = nn.Softmax(dim=1)
        self.save()

        return criterion, last_activation

    def get_label_encoder_criterion(self, multi_or_single: str, device=None):
        """
        call this function in place of get_label_encoder_criterion_for_model if the model is not available
        @type multi_or_single: str: can be 'multi' or 'single' for multi-class or binary classification
        """
        self.create_label_encoder(1 if multi_or_single == 'single' else 2)
        if multi_or_single == 'single':
            criterion = nn.BCELoss(reduction='mean')
            last_activation = nn.Sigmoid()
        elif multi_or_single == 'multi':
            criterion = nn.CrossEntropyLoss(weight=self.get_class_weight(True, device) if self.rebalance_method == 'class_weight' else None)
            last_activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"multi_or_single must be either 'multi' or 'single', but got {multi_or_single}")
        self.save()
        return criterion, last_activation

    def get_encoder_function(self):
        return self._encoder

    def save_to_path(self, path):
        """
        save the dataset to a given path
        @param path:
        @return:
        """
        copy_without_encoder = copy.deepcopy(self)
        copy_without_encoder._encoder = None
        try:
            pickle.dump(copy_without_encoder, open(path, 'wb'))
        except KeyboardInterrupt:
            pickle.dump(copy_without_encoder, open(self.filename, 'wb'))
            raise KeyboardInterrupt

    def save(self):
        """
        save the dataset to the default path
        @param path:
        @return:
        """
        copy_without_encoder = copy.deepcopy(self)
        copy_without_encoder._encoder = None
        try:
            pickle.dump(copy_without_encoder, open(self.filename, 'wb'))
        except KeyboardInterrupt:
            pickle.dump(copy_without_encoder, open(self.filename, 'wb'))
            raise KeyboardInterrupt

    def training_val_test_split_ordered_by_subject_run(self, n_folds, batch_size, val_size, test_size, random_seed=None, split_picks=None):
        """
        generate the sample indices for each fold and each batch, for train, val, and test
        consecutive batches have consecutive samples, for example, when the batch size is 3, and there are 120 samples for
        this subject & run
            batch0: [0, 40, 80]
            batch1: [1, 41, 81]
            batch2: [2, 42, 82]
            ...
            batch40: [39, 79, 119]

        the method go through each participant and each run, finds a random starting index for test and val. Then get
        consecutive samples for each test and val batches. The remaining samples are for training. So for each participant&run,
        the training data will have at most two points where it's not consecutive. That's where the val and test data are.
        All data partition (train, val, test) are consecutive within themselves.

        also note for each participant&run with n_sample samples, we generate up to < n_sample // batch_size > batches.
        so any residue samples are ignored. Using a smaller batch size will result in more batches and less residue samples.

        @param n_folds:
        @param batch_size:
        @param val_size:
        @param test_size:
        @param random_seed:
        @return:
        """
        np.random.seed(random_seed)


        subject_meta = self.physio_arrays[0].get_meta_info_by_name('subject_id')
        run_meta = self.physio_arrays[0].get_meta_info_by_name('run')

        all_sample_indices = np.arange(self.get_num_samples())
        # all_sample_indices = np.random.permutation(all_sample_indices)  # TODO testing with shuffled batches

        subject_run_samples = {(subject, run): all_sample_indices[np.logical_and(subject_meta==subject, run_meta==run)] for subject, run in itertools.product(np.unique(subject_meta), np.unique(run_meta))}

        test_batch_sample_indices = np.empty((0, batch_size), dtype=int)
        val_batch_sample_indices = [np.empty((0, batch_size), dtype=int) for i in range(n_folds)]
        train_batch_sample_indices = [np.empty((0, batch_size), dtype=int) for i in range(n_folds)]

        if split_picks:
            n_folds = len(split_picks['subjects'])
            self.training_val_split_indices = self.get_indices_from_picks(split_picks)
            for fold in range(n_folds):
                train_indices_fold = fill_batch_with_none_indices(self.training_val_split_indices[fold][0], batch_size)
                val_indices_fold = fill_batch_with_none_indices(self.training_val_split_indices[fold][1], batch_size)
                train_batch_sample_indices[fold] = train_indices_fold.reshape(batch_size, -1).T.astype(int)
                val_batch_sample_indices[fold] = val_indices_fold.reshape(batch_size, -1).T.astype(int)
        else:
            for (subject, run), sample_indices in subject_run_samples.items():
                n_batches = len(sample_indices) // batch_size
                if n_batches == 0:
                    warnings.warn(f"Subject {subject} run {run} has fewer samples than batch size. Ignored.")
                    continue
                # TODO this should be n_add = len(sample_indices) % (batch_size * n_batches)
                warnings.warn("Please check the TODO in the code.")
                n_add = batch_size - len(sample_indices) % (batch_size * n_batches)
                sample_indices = np.concatenate([sample_indices, [None] * n_add])
                n_batches = len(sample_indices) / batch_size
                assert n_batches.is_integer()
                n_batches = int(n_batches)

                n_test_batches = math.floor(test_size * n_batches)
                n_val_batches = math.floor(val_size * n_batches)
                if n_test_batches == 0 or n_val_batches == 0:
                    warnings.warn(f"Subject {subject} run {run} have too few samples to create enough batches for test and val.{n_batches =}. Ignored.")
                    # TODO maybe when this subject&run doesn't have enough samples, we can add it to the next subject&run
                    continue
                batch_indices = sample_indices[:n_batches * batch_size].reshape(batch_size, -1).T  # n_batches x batch_size
                print(f"Generated {n_batches} batches for subject {subject} run {run}. Last {len(sample_indices) - batch_size * n_batches} samples are ignored.")

                test_start_index = np.random.randint(0, n_batches - n_test_batches)
                test_batch_indices = np.arange(test_start_index, test_start_index + n_test_batches)
                test_batch_sample_indices = np.concatenate([test_batch_sample_indices, batch_indices[test_batch_indices]])

                for fold in range(n_folds):
                    val_start_index = np.random.choice([np.random.randint(0, test_start_index - n_val_batches)] if test_start_index > n_val_batches else [] +
                                                        [np.random.randint(test_start_index + n_test_batches, n_batches - n_val_batches)] if test_start_index + n_test_batches < n_batches - n_val_batches else [])
                    val_batch_indices = np.arange(val_start_index, val_start_index + n_val_batches)

                    val_batch_sample_indices[fold] = np.concatenate([val_batch_sample_indices[fold], batch_indices[val_batch_indices]])
                    train_batch_sample_indices[fold] = np.concatenate([train_batch_sample_indices[fold], np.delete(batch_indices, np.concatenate([val_batch_indices, test_batch_indices]), axis=0)])
        self.test_batch_sample_indices = np.array(test_batch_sample_indices)
        self.val_batch_sample_indices = np.array(val_batch_sample_indices)
        self.train_batch_sample_indices = np.array(train_batch_sample_indices)
        for i in range(n_folds):
            assert set(self.test_batch_sample_indices[self.test_batch_sample_indices != None]).intersection(set(self.train_batch_sample_indices[i][self.train_batch_sample_indices[i] != None])) == set(), "test and train is not disjoint"
            assert set(self.test_batch_sample_indices[self.test_batch_sample_indices != None]).intersection(set(self.val_batch_sample_indices[i][self.val_batch_sample_indices[i] != None])) == set(), "test and val is not disjoint"
            assert set(self.train_batch_sample_indices[i][self.train_batch_sample_indices[i] != None]).intersection(set(self.val_batch_sample_indices[i][self.val_batch_sample_indices[i] != None])) == set(), "train and val is not disjoint"
        self.save()

    def get_train_val_ordered_batch_iterator_fold(self, fold, device, shuffle_within_batches=False, encode_y=True, *args, **kwargs):
        """
        get a batch iterator for a specific fold
        @param fold:
        @param batch_size:
        @param batch_type:
        @param random_seed:
        @return:
        """
        assert self.val_batch_sample_indices is not None and self.train_batch_sample_indices is not None, \
            "Please call training_val_test_split_ordered_by_subject_run() first."
        if self.labels_array is not None:
            _labels  = self._encoder(self.labels_array) if encode_y else np.copy(self.labels_array)
            rtn = OrderedBatchIterator(self.physio_arrays, _labels, self.train_batch_sample_indices[fold], device, shuffle_within_batches), \
                    OrderedBatchIterator(self.physio_arrays, _labels, self.val_batch_sample_indices[fold], device, shuffle_within_batches),
            return rtn
        else:
            return OrderedBatchIterator(self.physio_arrays, None, self.train_batch_sample_indices[fold], device, shuffle_within_batches), \
                OrderedBatchIterator(self.physio_arrays, None, self.val_batch_sample_indices[fold], device, shuffle_within_batches)

    def get_test_ordered_batch_iterator(self, device, encode_y=True, shuffle_within_batches=False, *args, **kwargs):
        assert self.test_batch_sample_indices is not None, "Please call training_val_test_split_ordered_by_subject_run() first."
        if self.labels_array is not None:
            labels = self._encoder(self.labels_array) if encode_y else self.labels_array
            if not encode_y:  # check the labels are going from 0 to n_classes - 1
                assert np.all(np.unique(labels) == np.arange(len(self.event_names))), "labels are not going from 0 to n_classes - 1"
            return OrderedBatchIterator(self.physio_arrays, labels, self.test_batch_sample_indices, device, shuffle_within_batches=shuffle_within_batches)
        else:
            return OrderedBatchIterator(self.physio_arrays, None, self.test_batch_sample_indices, device, shuffle_within_batches=shuffle_within_batches)

    def get_encoded_labels(self):
        labels_encoded = None
        if self.labels_array is not None:
            labels_encoded = self._encoder(self.labels_array)
        return labels_encoded

    def check_correctness(self):
        '''
        check if the mmarray is correct.
        1. Check the batch is ordered.
        2. Check training, validation and test set do not overlap.
        This should be called before training start if verbose is set to True.
        @return:
        '''
        pass

    def has_split(self, use_ordered):
        if use_ordered:
            return self.train_batch_sample_indices is not None and self.val_batch_sample_indices is not None and self.test_batch_sample_indices is not None
        else:
            return self.train_indices is not None and self.test_indices is not None and self.training_val_split_indices is not None

    def create_split(self, use_ordered, n_folds, batch_size, val_size, test_size, random_seed,
                     split_picks=None, force_resplit=False, *args, **kwargs):
        """"
        use this function when you want to create a train, val, test split for mmarray, ordered or unordered
        It checks if the mmarray already has a split, if not, it will create a new split. It will also create a new split if force_resplit is True
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        if use_ordered:
            split_func = self.training_val_test_split_ordered_by_subject_run
            train_val_func = self.get_train_val_ordered_batch_iterator_fold
        else:
            split_func = self.train_test_val_split
            train_val_func = self.get_dataloader_fold

        if not self.has_split(use_ordered) or force_resplit:
            print( f"creating {'ordered' if use_ordered else 'unordered'} split for mmarray {n_folds} folds, and {len(self)} samples")
            split_func(n_folds, batch_size=batch_size, val_size=val_size, test_size=test_size, random_seed=random_seed,split_picks=split_picks)

    def get_test_dataloader(self, use_ordered, device, encode_y, *args, **kwargs):
        if use_ordered:
            return self.get_test_ordered_batch_iterator(device, encode_y, *args, **kwargs)
        else:
            test_set = self.get_test_set(use_ordered, convert_to_tensor=True, device=device)
            test_dataloader = DataLoader(test_set,  *args, **kwargs)
            return test_dataloader

    def get_train_val_loader(self, fold, use_ordered, device, encode_y, *args, **kwargs):
        if use_ordered:
            return self.get_train_val_ordered_batch_iterator_fold(fold, device, encode_y=encode_y, *args, **kwargs)
        else:
            return self.get_dataloader_fold(fold, device, encode_y, *args, **kwargs)

def load_mmarray(path):
    """
    load a mmarray from a path
    @param path:
    @return:
    """
    mmarray = pickle.load(open(path, 'rb'))
    return mmarray