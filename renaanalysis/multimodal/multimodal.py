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
from torch.utils.data import TensorDataset, DataLoader

from renaanalysis.learning.HT import HierarchicalTransformerContrastivePretrain, HierarchicalTransformer, \
    ContrastiveLoss
from renaanalysis.learning.MutiInputDataset import MultiInputDataset
from renaanalysis.multimodal.BatchIterator import OrderedBatchIterator
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
    def __init__(self, array: np.ndarray, meta_info: dict, sampling_rate: float, physio_type: str, is_rebalance_by_channel=False, dataset_name=''):
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
    def __init__(self, physio_arrays: List[PhysioArray], labels_array: np.ndarray =None, dataset_name: str='', event_viz_colors: dict=None, rebalance_method='SMOTE', filename=None):
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

    def get_dataloader_fold(self, fold_index, batch_size, is_rebalance_training=True, random_seed=None, device=None, task_name=TaskName.TrainClassifier, return_metainfo=False):
        """
        get the dataloader for a given fold
        this function must be called after training_val_split
        @param fold:
        @param random_seed:
        @return:
        """
        if self.rebalance_method == 'class_weight' and is_rebalance_training and self._encoder_object is not None and isinstance(self._encoder_object, LabelEncoder):
            warnings.warn("Using class_weight as rebalancing method while encoder is LabelEncoder because BCELoss can not apply class weights ")
        if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        # assert self.labels_array is not None, 'labels array must be provided to use rebalancing'
            assert self._encoder is not None, 'get_label_encoder_criterion_for_model must be called before get_rebalanced_dataloader_fold'
            training_indices, val_indices = self.training_val_split_indices[fold_index]
            x_train = []
            x_val = []
            y_train = self.labels_array[training_indices]
            y_val = self.labels_array[val_indices]

            labels = []
            for parray in self.physio_arrays:
                this_x_train, this_y_train = parray[training_indices], y_train
                if self.rebalance_method == 'SMOTE' and is_rebalance_training:
                    assert self.labels_array is not None, 'get_dataloader_fold: labels array must be provided to use rebalancing'
                    this_x_train, this_y_train = rebalance_classes(parray[training_indices], y_train, by_channel=parray.is_rebalance_by_channel, random_seed=random_seed)
                x_train.append(torch.Tensor(this_x_train).to(device))
                x_val.append(torch.Tensor(parray[val_indices]).to(device))

                labels.append(this_y_train)  # just for assertion

                # meta_info_train.append({name: values[training_indices] for name, values in parray.meta_info.items()})
                # meta_info_val.append({name: values[val_indices] for name, values in parray.meta_info.items()})
            meta_info_train = [{name: torch.Tensor(value[training_indices]).to(device) for name, value in darray.meta_info_encoded.items()} for darray in self.physio_arrays]
            meta_info_val = [{name: torch.Tensor(value[val_indices]).to(device) for name, value in darray.meta_info_encoded.items()} for darray in self.physio_arrays]

            meta_info_train = [v for d in meta_info_train for k, v in d.items()]
            meta_info_val = [v for d in meta_info_val for k, v in d.items()]

            n_train, n_val = len(x_train[0]), len(x_val[0])
            if len(meta_info_train) > 0 and len(meta_info_train[0]) != n_train and return_metainfo:
                warnings.warn("get_dataloader_fold: return_metainfo is not supported for SMOTE. The meta info will be duplicated using the first sample's")
                meta_info_train = [v[0].repeat(n_train,  *([1] * (len(v.shape)-1) ) ) for v in meta_info_train]
                meta_info_val = [v[0].repeat(n_val, *([1] * (len(v.shape)-1) ) ) for v in meta_info_val]

            assert np.all([label_set == labels[0] for label_set in labels])
            y_train = labels[0]
            y_train_encoded = self._encoder(y_train)
            y_val_encoded = self._encoder(y_val)
            y_train_encoded = torch.Tensor(y_train_encoded)
            y_val_encoded = torch.Tensor(y_val_encoded)

            if len(x_train) == 1:   # how many input modalities
                dataset_class = TensorDataset
            else:
                dataset_class = MultiInputDataset
            train_set = (*x_train, *meta_info_train, y_train_encoded ) if return_metainfo else (*x_train, y_train_encoded)
            val_set = (*x_val, *meta_info_val, y_val_encoded )if return_metainfo else (*x_val, y_val_encoded)
            train_dataset = dataset_class(*train_set)
            val_dataset = dataset_class(*val_set)
        elif task_name == TaskName.PreTrain:
            training_indices, val_indices = self.training_val_split_indices[fold_index]
            x_train = []
            x_val = []
            for parray in self.physio_arrays:
                x_train.append(torch.Tensor(parray[training_indices]).to(device))
                x_val.append(torch.Tensor(parray[val_indices]).to(device))
            if len(x_train) == 1:
                dataset_class = TensorDataset
                x_train = x_train[0]
                x_val = x_val[0]
            else:
                dataset_class = MultiInputDataset
            train_dataset = dataset_class(x_train)
            val_dataset = dataset_class(x_val)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader

    # def get_ordered_dataloader_fold(self, fold_index, batch_size, val_size, test_size, random_seed=None, device=None):


    def training_val_split(self, n_folds, val_size=0.1, random_seed=None):
        """
        split the train set into training and validation sets

        this function must be called after train_test_split
        @return:
        """
        assert self.train_indices is not None, 'train indices have not been set, please call train_test_split first'
        self.training_val_split_indices = []
        if self.labels_array is not None:
            skf = StratifiedShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
            for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array[self.train_indices], self.labels_array[self.train_indices])):
                self.training_val_split_indices.append((train, val))
        else:
            skf = ShuffleSplit(test_size=val_size, n_splits=n_folds, random_state=random_seed)
            for f_index, (train, val) in enumerate(skf.split(self.physio_arrays[0].array[self.train_indices])):
                self.training_val_split_indices.append((train, val))
        self.save()


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
            self.train_indices, self.test_indices = train_test_split(list(range(self.physio_arrays[0].array.shape[0])), test_size=test_size, random_state=random_seed, stratify=self.labels_array)
        self.save()

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

    def get_test_set(self, encode_y=False, convert_to_tensor=False, device=None, return_metainfo=False):
        """
        get the test set
        @return:
        """
        assert self.test_indices is not None, 'test indices have not been set, please call train_test_split first'
        x_test = []
        for parray in self.physio_arrays:
            is_pca_ica_preprocessed = 'pca' in parray.data_processor.keys() or 'ica' in parray.data_processor.keys()
            if is_pca_ica_preprocessed:
                print("\033[93m  {}\033[00m".format('test set is pca or ica preprocessed, make sure preprocessing is needed for this model'))
            else:
                print("\033[93m  {}\033[00m".format('test set is not pca or ica preprocessed, make sure preprocessing is not needed for this model'))
            if convert_to_tensor:
                x_test.append(torch.Tensor(parray[self.test_indices]).to(device))
            else:
                x_test.append(parray[self.test_indices])


        if self.labels_array is not None:
            y_test = self.labels_array[self.test_indices]
            if encode_y:
                y_test = self._encoder(y_test)
            if convert_to_tensor:
                y_test = torch.Tensor(y_test).to(device)
        else:
            warnings.warn('labels array is None, make sure label is not needed for this model')
            y_test = None

        meta_info = [{name: value[self.test_indices] for name, value in darray.meta_info_encoded.items()} for darray in self.physio_arrays]
        meta_info = {k: v for d in meta_info for k, v in d.items()}
        if convert_to_tensor:
            meta_info = {k: torch.Tensor(v).to(device) for k, v in meta_info.items()}
        test_set = (*x_test, *list(meta_info.values()), y_test) if return_metainfo else (*x_test, y_test)

        return test_set

    def get_test_dataloader(self, batch_size, encode_y=False, device=None, return_metainfo=False):
        test_set = self.get_test_set(encode_y=encode_y, convert_to_tensor=True, device=device, return_metainfo=return_metainfo)
        if len(self.physio_arrays) == 1:
            dataset_class = TensorDataset
        else:
            dataset_class = MultiInputDataset
        test_dataset = dataset_class(*test_set)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        return test_dataloader

    def get_random_sample(self, convert_to_tensor=False, device=None, include_metainfo=False):
        """
        @return: a random sample from each of the physio arrays
        """
        random_sample_index = np.random.randint(0, len(self.physio_arrays[0]))
        rtn = [(parray[random_sample_index][None, :]) for parray in self.physio_arrays]
        rtn = [torch.tensor(r, dtype=torch.float32, device=device) for r in rtn] if convert_to_tensor else rtn
        rtn = rtn if len(rtn) > 1 else rtn[0]

        meta_info = None
        if include_metainfo:
            meta_info = [parray.get_meta_info(random_sample_index, encoded=True) for parray in self.physio_arrays]
            meta_info = [[torch.tensor([value], device=device) for name, value in m.items()] for m in meta_info] if convert_to_tensor else meta_info
            meta_info = list(itertools.chain.from_iterable(meta_info))
        if include_metainfo:
            return rtn, *meta_info
        else:
            return rtn

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


    def get_label_encoder_criterion_for_model(self, model, device=None, include_metainfo=False):
        """
        this function must be called

        @param model:
        @param device:
        @param class_weights:
        @return:
        """
        assert not isinstance(model, HierarchicalTransformerContrastivePretrain), "Model is HierarchicalTransformerContrastivePretrain, which does not have label encoder, please use function get_critierion_for_pretrain"
        rand_input = self.get_random_sample(convert_to_tensor=True, device=device, include_metainfo=include_metainfo) if include_metainfo else self.get_random_sample(convert_to_tensor=True, device=device)
        with torch.no_grad():
            model.eval()
            rand_input= rand_input if isinstance(rand_input, tuple) else (rand_input,)
            output_shape = model.to(device)(*rand_input).shape[1]

        self.create_label_encoder(output_shape)
        if output_shape == 1:
            criterion = nn.BCELoss(reduction='mean')
            last_activation = nn.Sigmoid()
        else:
            criterion = nn.CrossEntropyLoss(weight=self.get_class_weight(True, device) if self.rebalance_method == 'class_weight' else None)
            last_activation = nn.Softmax(dim=1)
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

    def training_val_test_split_ordered_by_subject_run(self, n_folds, batch_size, val_size, test_size, random_seed=None):
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

        for (subject, run), sample_indices in subject_run_samples.items():
            n_batches = len(sample_indices) // batch_size
            n_add = batch_size - len(sample_indices) % (batch_size * n_batches)
            sample_indices = np.concatenate([sample_indices, [None] * n_add])
            n_batches = len(sample_indices) / batch_size
            assert n_batches.is_integer()
            n_batches = int(n_batches)
            if n_batches == 0:
                warnings.warn(f"Subject {subject} run {run} has less samples than batch size. Ignored.")
                continue
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
        self.save()

    def get_train_val_ordered_batch_iterator_fold(self, fold, device, return_metainfo=False):
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

        labels_encoded = self._encoder(self.labels_array)

        return OrderedBatchIterator(self.physio_arrays, labels_encoded, self.train_batch_sample_indices[fold], device, return_metainfo), \
            OrderedBatchIterator(self.physio_arrays, labels_encoded, self.val_batch_sample_indices[fold], device, return_metainfo)

    def get_test_ordered_batch_iterator(self, device, return_metainfo=False):
        assert self.test_batch_sample_indices is not None, "Please call training_val_test_split_ordered_by_subject_run() first."
        labels_encoded = self._encoder(self.labels_array)
        return OrderedBatchIterator(self.physio_arrays, labels_encoded, self.test_batch_sample_indices, device, return_metainfo)
    # def traning_val_test_split_ordered(self, n_folds, batch_size, val_size, test_size, random_seed=None):
    #     n_batches = self.get_num_samples() // batch_size
    #     test_n_batches = math.floor(test_size * n_batches)
    #     test_start_index = np.random.randint(0, n_batches - test_n_batches)
    #     test_batch_indices = np.arange(test_start_index, test_start_index + test_n_batches)
    #
    #     val_n_batches = math.floor(val_size * n_batches)
    #     val_start_index = np.random.choice([np.random.randint(0, test_start_index - val_n_batches) if test_start_index > val_n_batches else [], np.random.randint(test_start_index + test_n_batches, n_batches - val_n_batches)])
    #     val_batch_indices = np.arange(val_start_index, val_start_index + val_n_batches)
    #
    #     batch_indices = np.arange(n_batches * batch_size).reshape(batch_size, -1).T  # n_batches x batch_size
    #
    #     val_sample_indices = batch_indices[val_batch_indices]
        # self.physio_arrays[0][]


def load_mmarray(path):
    """
    load a mmarray from a path
    @param path:
    @return:
    """
    mmarray = pickle.load(open(path, 'rb'))
    return mmarray