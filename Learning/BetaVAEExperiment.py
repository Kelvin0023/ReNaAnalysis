from Learning.base import BaseVAE
from Learning.beta_vae import BetaVAE
import numpy as np

# load data
data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP.npy'
label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epoch_labels_pupil_raw_condition_RSVP.npy'
X = np.load(data_path)
y = np.load(label_path)

# define the training
def training_step(self, batch, batch_idx, optimizer_idx=0):
    real_img, labels = batch
    self.curr_device = real_img.device

    results = self.forward(real_img, labels=labels)
    train_loss = self.BRNN_model.loss_function(*results,
                                               M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                               optimizer_idx=optimizer_idx,
                                               batch_idx=batch_idx)

    self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

    return train_loss['loss']

model:BaseVAE = BetaVAE
