import warnings
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from renaanalysis.multimodal.PhysioArray import PhysioArray
from renaanalysis.utils.TorchUtils import batch_to_tensor
from renaanalysis.utils.data_utils import check_and_merge_dicts, rebalance_classes, check_arrays_equal


class MultiModalDataset(Dataset):
    def __init__(self, physio_arrays: List[PhysioArray], labels=None, use_meta_info=True, indices=None):

        """

        @param physio_arrays:
        @param labels:
        @param use_meta_info:
        @param indices:  can be passed in to select a subset of the data, convenient for selecting subset for train, val, test
        It will use all data if None
        """
        self.physio_type_rebalance = {parray.physio_type: parray.is_rebalance_by_channel for parray in physio_arrays}
        data: Dict[str, np.ndarray] = {parray.physio_type: parray.get_array() for parray in physio_arrays}
        meta_info: Dict[str, dict] = {parray.physio_type: parray.meta_info_encoded for parray in physio_arrays}

        assert len(data) > 0, "data must be a non-empty dict."
        assert len({(n_data_samples := len(d)) for modality, d in data.items()}) == 1, "data must have the same number of samples."

        self.data = {physio_type: (d[indices] if indices is not None else d) for physio_type, d in data.items()}
        self.meta_info = {}
        merged_meta_info = check_and_merge_dicts(meta_info)  # : meta_info_name (str) -> meta_info_value for each samples (array/tensor)
        if len(merged_meta_info) > 0:
            # check the list of meta info dicts doesn't have duplicate keys
            # check data has the same number of samples
            # check meta_info has the same number of samples as data
            assert len({(n_meta_samples := len(d)) for meta_name, d in merged_meta_info.items()}) == 1, "meta_info must have the same number of samples."
            assert n_data_samples == n_meta_samples, "data and meta_info must have the same number of samples."
            self.meta_info = {meta_name: (d[indices] if indices is not None else d) for meta_name, d in merged_meta_info.items()}

        self.labels = labels
        if labels is not None:
            assert len(labels) == n_data_samples, "labels must have the same number of samples as data and meta_info."
            self.labels = labels[indices] if indices is not None else labels

        self.use_meta_info = use_meta_info
        self.n_samples = n_data_samples if indices is None else len(indices)

        # keep np copies
        self.data_np = {physio_type: d.copy() for physio_type, d in self.data.items()}
        self.meta_info_np = {meta_name: d.copy() for meta_name, d in self.meta_info.items()}
        self.labels_np = self.labels.copy() if self.labels is not None else None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        meta_info = {meta_name: d[idx] for meta_name, d in self.meta_info.items()} if self.use_meta_info else {}
        if self.labels is not None:
            return {**{physio_type: d[idx] for physio_type, d in self.data.items()}, **meta_info, 'y': self.labels[idx]}
        return {**{physio_type: d[idx] for physio_type, d in self.data.items()}, **meta_info}

    def get_meta_info(self, idx):
        return {meta_name: d[idx] for meta_name, d in self.meta_info.items()} if self.use_meta_info else {}

    def get_data(self, idx):
        return {physio_type: d[idx] for physio_type, d in self.data.items()}

    def get_labels(self, idx):
        return self.labels[idx]

    def get_rebalanced_set(self, random_seed, encoder):
        """

        @param random_seed:
        @param encoder: needed because labels will not be one-hot encoded after rebalancing
        @return:
        """
        assert self.labels is not None, 'MultiModalDataset: get_rebalanced_set: labels array must be provided to use rebalancing'
        # rebalance by each modality
        self.data = {physio_type: rebalance_classes(d, self.labels, by_channel=self.physio_type_rebalance[physio_type], random_seed=random_seed) for physio_type, d in self.data.items()}
        self.labels = [l for physio_type, (d, l) in self.data.items()]
        # check all the labels are equal after rebalancing
        assert check_arrays_equal(self.labels), 'MultiModalDataset: get_rebalanced_set: after rebalancing, the labels are not equal across all modalities'
        self.labels = encoder(self.labels[0])
        self.data = {physio_type: d for physio_type, (d, l) in self.data.items()}
        self.n_samples = len(self.labels)

        if self.use_meta_info:
            # get meta info by duplicating the first sample's
            warnings.warn("MultiModalDataset: get_rebalanced_set: use_meta_info is not supported for rebalancing. The meta info will be duplicated using the first sample's")
            for meta_name, meta in self.meta_info.items():
                self.meta_info[meta_name] = np.repeat(meta[0], self.n_samples,)
            self.meta_info = {meta_name: np.tile(d[0], (len(self), *([1] * (len(d.shape) - 1)))) for meta_name, d in self.meta_info.items()}


    def to_tensor(self, device):
        self.data = batch_to_tensor(self.data, device)
        self.meta_info = batch_to_tensor(self.meta_info, device)
        if self.labels is not None:
            label_dtype = torch.long if len(self.labels.shape) == 1 else torch.float
            self.labels = torch.tensor(self.labels, device=device).to(label_dtype)
