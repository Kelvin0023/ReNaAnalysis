import numpy as np
import torch


class ordered_batch_iterator:
    def __init__(self, data_arrays, labels, batch_sample_indices, device, return_metainfo=False):
        assert len(batch_sample_indices.shape) == 2, "batch_sample_indices must be a 2D array."
        # check the batch is ordered
        assert np.all(np.diff(batch_sample_indices, axis=0) > 0), "batch_sample_indices is not ordered."
        self.data_arrays = data_arrays
        self.dataset = data_arrays[0]  # use the first array to get the number of samples
        self.labels = labels

        # check the max sample index is no greater than the number of samples
        assert np.max(batch_sample_indices) < len(self.dataset)

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
        if self.batch_index >= self.n_batches:
            raise StopIteration
        this_batch_sample_indices = self.batch_sample_indices[self.batch_index]
        batch = [torch.Tensor(darray.array[this_batch_sample_indices]).to(self.device) for darray in self.data_arrays]
        batch = batch[0] if len(batch) == 1 else batch
        self.batch_index += 1
        labels = torch.Tensor(self.labels[this_batch_sample_indices]).to(self.device)
        if self.return_metainfo:
            meta_info = [{name: torch.Tensor(value[this_batch_sample_indices]).to(self.device) for name, value in darray.meta_info_encoded.items()} for darray in self.data_arrays]
            meta_info = meta_info[0] if len(meta_info) == 1 else meta_info
            return (batch, meta_info), labels
        else:
            return (batch), labels