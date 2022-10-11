import numpy as np
import cv2
import torch


def prepare_image_for_sim_score(img):
    img = np.moveaxis(cv2.resize(img, (64, 64)), -1, 0)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    return img