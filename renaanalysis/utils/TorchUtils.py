from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn


def prepare_image_for_sim_score(img):
    img = np.moveaxis(cv2.resize(img, (64, 64)), -1, 0)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    return img


def batch_to_tensor(batch: Dict[str, np.ndarray], device, dtype=torch.float32):
    """
    Convert a batch dict of numpy arrays to a batch of torch tensors.
    @param batch:
    @param device:
    @return:
    """
    rtn = {name: torch.tensor(x, dtype=dtype, device=device) for name, x in batch.items()}
    return rtn

def get_output_size(net: nn.Module, input_shape):
    """

    @param net: the module to get the output size
    @param input_shape: the input size without batch size
    """
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        output = net(x)
        return output.shape[1:]
