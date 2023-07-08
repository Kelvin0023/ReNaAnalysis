import numpy as np

from renaanalysis.utils.data_utils import z_norm_by_trial, compute_pca_ica


class PhysioArray:
    """
    when rebalance is applied, is this array rebalanced by channel or not
    @attribute sampling_rate
    @attribute physio_type
    @attribute is_rebalance_by_channel
    """
    def __init__(self, array: np.ndarray, sampling_rate: float, physio_type: str, is_rebalance_by_channel=False, dataset_name=''):
        self.array = array
        self.sampling_rate = sampling_rate
        self.physio_type = physio_type
        self.is_rebalance_by_channel = is_rebalance_by_channel
        self.data_processor = dict()
        self.dataset_name = dataset_name
    def array(self):
        return self.array
    def __getitem__(self, item):
        return self.array[item]

    def __len__(self):
        return len(self.array)

    def __str__(self):
        data_preprocessor_str = '-'.join(self.data_processor.keys())
        return f'{self.dataset_name}_{self.physio_type}_{data_preprocessor_str}'

    def apply_znorm_by_trial(self):
        self.array = z_norm_by_trial(self.array)
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
        self.array, self.data_processor['pca'], self.data_processor['ica'] = compute_pca_ica(self.array, n_top_components)
        self.data_processor['pca_ica_components'] = n_top_components

class MultiModalArrays():
    def __init__(self, physio_arrays: list, labels_array=None, dataset_name='', event_viz_colors=None):
        self.physio_arrays = physio_arrays
        self.labels_array = labels_array
        self.dataset_name = dataset_name
        self.event_viz_colors = event_viz_colors
        self._physio_types = [parray.physio_type for parray in physio_arrays]
        self._physio_types_arrays = dict(zip(self._physio_types, self.physio_arrays))

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


