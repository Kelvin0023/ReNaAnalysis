import lpips
import numpy as np
import torch
import cv2

image0_path = 'C:/Users/Lab-User/Dropbox/ReNa/data/ReNaPilot-2022Spring/3-5-2022/0/ReNaUnityCameraCapture_03-05-2022-13-42-32/32473.png'
image1_path = 'C:/Users/Lab-User/Dropbox/ReNa/data/ReNaPilot-2022Spring/3-5-2022/0/ReNaUnityCameraCapture_03-05-2022-13-42-32/32816.png'
image0 = cv2.imread(image0_path)
image1 = cv2.imread(image1_path)

img0 = np.moveaxis(cv2.resize(image0, (64, 64)), -1, 0)
img1= np.moveaxis(cv2.resize(image1, (64, 64)), -1, 0)

img0 = np.expand_dims(img0, axis=0)
img1 = np.expand_dims(img1, axis=0)

img0 = torch.Tensor(img0)
img1 = torch.Tensor(img1)

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

d = loss_fn_alex(img0, img1)
