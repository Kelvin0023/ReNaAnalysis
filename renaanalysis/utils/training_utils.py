import numpy as np
import torch


def count_standard_error(true_label, pred_label):
    count = 0
    for elem1, elem2 in zip(true_label, pred_label):
        if elem1 == 0 and elem2 == 1:
            count += 1
    return count

def count_target_error(true_label, pred_label):
    count = 0
    for elem1, elem2 in zip(true_label, pred_label):
        if elem1 == 1 and elem2 == 0:
            count += 1
    return count

def get_class_weight(labels):
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
    if len(labels.shape) == 2:  # if is onehot encoded
        unique_classes, counts = torch.unique(labels, return_counts=True, dim=0)
        counts = torch.flip(counts, dims=[0])  # refer to docstring
    elif len(labels.shape) == 1:
        unique_classes, counts = torch.unique(labels, return_counts=True)
    else:
        raise ValueError("encoded labels should be either 1d or 2d array")
    if len(counts) == 1:  # when there is only one class in the dataset
        return None
    class_proportions = counts / len(labels)
    class_weights = 1 / class_proportions
    return class_weights  # reverse the class weights because