import datetime
import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from renaanalysis.learning.train import eval_model, preprocess_model_data
from renaanalysis.params.params import *
from renaanalysis.utils.dataset_utils import get_auditory_oddball_samples

# test domain
def buffered_arange(max, device="cpu"):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor().to(device)
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]
class Wav2VecPredictionsModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        prediction_steps,
        n_negatives,
        cross_sample_negatives,
        sample_distance,
        dropout,
        offset,
        balanced_classes,
        infonce,
    ):
        super().__init__()

        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = sample_distance
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.offset = offset
        self.balanced_classes = balanced_classes
        self.infonce = infonce

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT

        return negs

    def forward(self, x, y):

        x = x.unsqueeze(-1)
        x = self.project_to_steps(x)  # BxCxTxS
        x = self.dropout(x)

        negatives = self.sample_negatives(y)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)  # Copies x B x C x T

        copies = targets.size(0)
        bsz, dim, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)

        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )
        if self.infonce:
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            labels = torch.zeros_like(predictions)
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        start = end = 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            if self.infonce:
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
            else:
                pos_num = (end - start) // copies
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                labels[start : start + pos_num] = 1.0
                if weights is not None:
                    weights[start : start + pos_num] = 1.0
            start = end
        assert end == predictions.numel(), "{} != {}".format(end, predictions.numel())

        if self.infonce:
            predictions = predictions.view(-1, copies)
        else:
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels

pred = Wav2VecPredictionsModel(1, 1, 1, 20, 5, None, 0.5, 0, False, True)
y = np.random.random((32, 64, 180))
y = torch.Tensor(y)
pred.sample_negatives(y)

result_path = 'results/model_performances_self_sup'
# analysis parameters ######################################################################################
eeg_resample_rate = 200
reject = 'auto'
bids_root = 'D:/Dataset/auditory_oddball'
event_names = ["standard", "oddball_with_reponse"]
colors = {
    "standard": "red",
    "oddball_with_reponse": "green"
}
picks = 'eeg'
# models = ['HT', 'HDCA', 'EEGCNN']
models = ['HT-pca-ica-sesup']
n_folds = 6
ht_lr = 1e-3
ht_l2 = 1e-5

reload_saved_samples = False
# start of the main block ######################################################

torch.manual_seed(random_seed)
np.random.seed(random_seed)

x, y = get_auditory_oddball_samples(bids_root, export_data_root, reload_saved_samples, event_names, picks, reject, eeg_resample_rate, colors)
with open(os.path.join('HT_sesup/','x_raw.p'), 'wb') as f:
    pickle.dump(x, f)
with open(os.path.join('HT_sesup/','y_raw.p'), 'wb') as f:
    pickle.dump(y, f)

event_ids = {'standard': 1, 'oddball_with_reponse': 7}
baseline = (-0.1, 0.0)
ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EOG00', 'EOG01']
eeg_picks = ['Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'AFz', 'Fz', 'FCz', 'Cz']
info = mne.create_info(ch_names=ch_names, sfreq=256)
# epochs = mne.EpochsArray(data, info)

x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed = preprocess_model_data(x, None)
with open('data/x_pca_ica.p', "wb") as file:
    pickle.dump(x_eeg_pca_ica, file)
with open('data/x_znormed.p', "wb") as file:
    pickle.dump(x_eeg_znormed, file)

results = dict()
now = datetime.datetime.now()
datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")
result_path = result_path + datetime_string

for m in models:
    m_results, training_histories = eval_model(x, None, y, event_names, model_name=m, exg_resample_rate=eeg_resample_rate, n_folds=n_folds, ht_lr=ht_lr, ht_l2=ht_l2, eeg_montage=eeg_montage,
                           x_eeg_znormed=x_eeg_znormed, x_eeg_pca_ica=x_eeg_pca_ica, x_pupil_znormed=x_pupil_znormed,
                           test_name=f"auditory_oddball_{m}_{datetime_string}")
    results = {**m_results, **results}
    pickle.dump(results, open(result_path, 'wb'))
    pickle.dump(training_histories, open(result_path + f'{m}_training_history', 'wb'))