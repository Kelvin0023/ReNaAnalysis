import os
import pickle

from renaanalysis.utils.dataset_utils import get_dataset


def model_comparison(train_params, model_class_params, datasets, export_data_root, is_regenerate_epochs):
    """

    @param train_params:
    @param model_class_params:
    @param datasets: dictionary of dataset name, data loading parameters
    @param n_folds:
    @param is_regenerate_epochs:
    @return:
    """
    for dataset_name, dataset_params in datasets.items():
        mmarray_fn = f'{dataset_name}_mmarray.p'
        mmarray_path = os.path.join(export_data_root, mmarray_fn)
        dataset_params['filename'] = mmarray_path
        if is_regenerate_epochs or not os.path.exists(mmarray_path):
            mmarray = get_dataset(**dataset_params)
            mmarray.save()
        else:
            mmarray = pickle.load(open(mmarray_path, 'rb'))
