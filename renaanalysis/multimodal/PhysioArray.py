import numpy as np
from sklearn.preprocessing import LabelEncoder

from renaanalysis.multimodal.ChannelSpace import create_discretize_channel_space
from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica, z_norm_by_subject_run


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
    def __init__(self, array: np.ndarray, meta_info: dict, sampling_rate: float, physio_type: str, is_rebalance_by_channel=False, dataset_name='', info=None, ch_names=None):
        assert np.all(array.shape[0] == np.array([len(m) for m in meta_info.values()])), 'all metainfo in a physio array must have the same number of trials/epochs'
        self.array = array
        self.meta_info = meta_info

        self.meta_info_encoders = dict()
        self.meta_info_encoded = dict()
        self.encode_meta_info()
        self.ch_names = ch_names

        self.sampling_rate = sampling_rate
        self.physio_type = physio_type
        self.is_rebalance_by_channel = is_rebalance_by_channel
        self.data_processor = dict()
        self.dataset_name = dataset_name

        self.array_preprocessed = None

        self.info = info

        if self.physio_type == 'eeg':
            create_discretize_channel_space(self)

    def __getitem__(self, item):
        if self.data_processor is not None:
            return self.array_preprocessed[item]
        else:
            return self.array[item]

    def get_array(self):
        if self.data_processor is not None:
            return self.array_preprocessed
        else:
            return self.array

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
        self.data_processor['znorm_by_trial'] = True

    def apply_znorm_by_run(self):
        self.array_preprocessed = z_norm_by_subject_run(self)
        self.data_processor['znorm_by_run'] = True

    def apply_znorm_global(self):
        self.array_preprocessed = self.array.copy()
        self.array_preprocessed -= self.array_preprocessed.mean()
        self.array_preprocessed /= self.array_preprocessed.std()
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
