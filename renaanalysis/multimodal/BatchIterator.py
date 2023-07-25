import copy

import numpy as np
import torch


def check_batch_is_ordered(batch_sample_indices):
    indices =  copy.copy(batch_sample_indices)
    indices_replaced_none = copy.copy(batch_sample_indices)
    indices_replaced_none[indices_replaced_none == None] = 0

    indices_diff = np.diff(indices_replaced_none, axis=0, prepend=0)
    indices_diff[indices == None] = 1
    assert np.all(indices_diff >= 0), "batch_sample_indices is not ordered."


class OrderedBatchIterator:
    def __init__(self, data_arrays, labels, batch_sample_indices, device, return_metainfo=False):
        no_none_batch_sample_indices = np.array(batch_sample_indices[batch_sample_indices != None], dtype=int)
        assert len(batch_sample_indices.shape) == 2, "batch_sample_indices must be a 2D array."
        # check the batch is ordered
        check_batch_is_ordered(batch_sample_indices)
        self.data_arrays = data_arrays
        self.dataset = data_arrays[0][np.array(no_none_batch_sample_indices, dtype=int)]  # use the first array to get the number of samples
        self.labels = labels

        # check the max sample index is no greater than the number of samples
        assert np.max(no_none_batch_sample_indices) < len(self.data_arrays[0])

        self.batch_sample_indices = batch_sample_indices
        self.batch_size = batch_sample_indices.shape[1]
        self.n_batches = batch_sample_indices.shape[0]
        self.return_metainfo = return_metainfo
        self.device = device

        self.batch_index = 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        """
        When self.data_arrays is a list of PhysioArray's. The default getitem method in PhysioArray will get the
        preprocessed data.
        @return:
        """
        if self.batch_index >= self.n_batches:
            self.batch_index = 0
            raise StopIteration
        this_batch_sample_indices = self.batch_sample_indices[self.batch_index]
        this_batch_sample_indices = np.array(this_batch_sample_indices[this_batch_sample_indices != None], dtype=int)
        self.batch_index += 1

        batch = [torch.Tensor(darray[this_batch_sample_indices]).to(self.device) for darray in self.data_arrays]
        batch = batch[0] if len(batch) == 1 else batch
        labels = torch.Tensor(self.labels[this_batch_sample_indices]).to(self.device)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        if self.return_metainfo:
            meta_info = [[torch.Tensor(value[this_batch_sample_indices]).to(self.device) for name, value in darray.meta_info_encoded.items()] for darray in self.data_arrays]
            meta_info = meta_info[0] if len(meta_info) == 1 else meta_info
            return batch, * meta_info, labels
        else:
            return batch, labels