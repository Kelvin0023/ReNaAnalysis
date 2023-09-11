
import os
import math

import mne
import numpy as np
import scipy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.utils.data import TensorDataset

# from renaanalysis.learning.HT_viz import ht_eeg_viz_dataloader_batch
from renaanalysis.utils.utils import visualize_eeg_epochs
from torch import optim, nn
from torch.optim import lr_scheduler
from sklearn import preprocessing
from renaanalysis.utils.data_utils import z_norm_by_trial
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor

from renaanalysis.utils.viz_utils import viz_confusion_matrix

# data_root = r'C:\Users\apoca\Downloads'
data_root = r'D:\Dropbox\Dropbox\EEGDatasets\BCICompetitionIV2a'
# data_root = r'D:/Dataset/BCICIV_2a/'

subject = 'A03'
eeg_channels = {'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',  'CP3', 'CP1', 'CPz',  'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'}
srate = 250
montage = mne.channels.make_standard_montage('standard_1020')
lowcut_eeg = 4
highcut_eeg = 40
n_jobs = 1
epoch_tmin = 0.
epoch_tmax = 4.
event_id = {'left': 1, 'right': 2, 'foot': 3, 'tongue': 4}
event_viz_colors = {'left': 'red', 'right': 'blue', 'foot': 'green', 'tongue': 'yellow'}


def load_mat_sessions(file_path, figure_title):
    this_epoch_tmax = epoch_tmax - 1 / srate
    data = scipy.io.loadmat(file_path)['data']
    all_epochs = []
    for session_i in range(data.shape[1]):
        ys = data[0, session_i]['y'][0, 0]
        if len(ys) == 0:
            print(f"session {session_i} of {data.shape[1]} does not have labels, skipping")
            continue
        # the last three channels are eog, so we skip them
        eeg = data[0, session_i]['X'][0, 0][:, :-3]
        trial_start_indices = data[0, session_i]['trial'][0, 0]
        # create mne data structure
        info = mne.create_info(ch_names=list(eeg_channels), sfreq=srate, ch_types='eeg')
        raw = mne.io.RawArray(eeg.T, info)
        raw.set_montage(montage)
        # raw, _ = mne.set_eeg_reference(raw, 'average', projection=False)
        # raw = raw.filter(l_freq=lowcut_eeg, h_freq=highcut_eeg, n_jobs=n_jobs, picks='eeg')

        # create events array from trial start indices
        events = np.array(np.concatenate([trial_start_indices, np.zeros((len(trial_start_indices), 1)), ys], axis=1),
                          dtype=int)
        epoch = mne.Epochs(raw, events, event_id=event_id, tmin=epoch_tmin, tmax=this_epoch_tmax,
                           baseline=(epoch_tmin, epoch_tmin + (epoch_tmax - epoch_tmin) * 0.1), preload=True,
                           event_repeated='drop')
        all_epochs.append(epoch)
    epochs = mne.concatenate_epochs(all_epochs)
    visualize_eeg_epochs(epochs, event_id, event_viz_colors, tmin_eeg_viz=epoch_tmin, tmax_eeg_viz=epoch_tmax,
                         eeg_picks=eeg_channels, title=figure_title)
    return epochs


from renaanalysis.utils.dataset_utils import get_BCI_montage


def load_gdf_sessions(train_file_path, val_file_path, figure_title):
    event_viz_colors = {'769': 'red', '770': 'blue', '771': 'green', '772': 'yellow'}
    channel_mapping = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
                       'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
                       'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2',
                       'EEG-13': 'CP4', 'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'}
    kept_channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1',
                     'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    event_id_mapping = {'769': 0, '770': 1, '771': 2, '772': 3, '783': 7}

    this_epoch_tmax = epoch_tmax - 1 / srate
    raw_train = mne.io.read_raw_gdf(train_file_path, preload=True)
    # raw_train, _ = mne.set_eeg_reference(raw_train, 'average', projection=False)
    mont1020 = get_BCI_montage('standard_1020', picks=kept_channels)

    raw_val = mne.io.read_raw_gdf(val_file_path, preload=True)
    # raw_val, _ = mne.set_eeg_reference(raw_val, 'average', projection=False)

    raw = mne.concatenate_raws([raw_train, raw_val])
    mne.rename_channels(raw.info, channel_mapping)

    raw.drop_channels(
        ['EOG-left', 'EOG-central', 'EOG-right'])  # otherwise the channel names are not consistent with montage
    raw.set_montage(mont1020)
    events, event_id = mne.events_from_annotations(raw, event_id=event_id_mapping)

    data = mne.Epochs(raw, events, event_id=event_id, tmin=epoch_tmin, tmax=this_epoch_tmax,
                      baseline=(epoch_tmin, epoch_tmin + (epoch_tmax - epoch_tmin) * 0.1), preload=True,
                      event_repeated='drop')
    # add the labels to the val epochs
    val_len = np.sum(events[:, -1] == event_id_mapping['783'])
    true_label_path_eval = os.path.join(data_root, 'true_labels', f'A03E.mat')
    true_label_eval = scipy.io.loadmat(true_label_path_eval)

    true_label_path_train = os.path.join(data_root, 'true_labels', f'A03T.mat')
    true_label_train = scipy.io.loadmat(true_label_path_train)
    assert np.all(
        data.events[:, -1][events[:, -1] != event_id_mapping['783']] == (true_label_train['classlabel'] - 1).squeeze(
            axis=-1)), f"Labels don't match for subject train"

    # data.events[:, -1] = (true_label['classlabel'] - 1).squeeze(axis=-1)
    # data.event_id = {'769': 0, '770': 1, '771': 2, '772': 3}
    assert np.all(np.argwhere(events[:, -1] == event_id_mapping['783'])[:, 0] == np.arange(288,
                                                                                           len(data))), f"Labels don't match for eval"
    data.events[:, -1][events[:, -1] == event_id_mapping['783']] = (true_label_eval['classlabel'] - 1).squeeze(axis=-1)
    data.event_id.pop('783')
    # visualize_eeg_epochs(data, event_id, event_viz_colors, tmin_eeg_viz=epoch_tmin, tmax_eeg_viz=epoch_tmax,
    #                      eeg_picks=eeg_channels, title=figure_title)
    return data[:val_len], data[val_len:]

# train_path = os.path.join(data_root, f'{subject}T.mat')
# eval_path = os.path.join(data_root, f'{subject}E.mat')
#
# train_epochs = load_mat_sessions(train_path, f'{subject} Train')
# eval_epochs = load_mat_sessions(eval_path, f'{subject} Eval')
#
train_path = os.path.join(data_root, f'{subject}T.gdf')
eval_path = os.path.join(data_root, f'{subject}E.gdf')

train_epochs, eval_epochs= load_gdf_sessions(train_path, eval_path, f'{subject} Train')


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            # nn.Linear(1600, 256),
            # nn.Linear(39040, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

# EEGCNN ##############################################################################
class EEGCNN(nn.Module):
    def __init__(self, in_shape, num_classes, num_filters=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_shape[1], num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters, num_filters, 5),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # with torch.no_grad():
        #     cnn_flattened_size = self.conv(torch.rand(in_shape)).shape[1]

        out_size = math.floor((math.floor((math.floor((in_shape[-1] -6)/2+1) -6)/2+1) - 6)/2 + 1) * num_filters

        self.fcs = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, input):
        x = self.conv(input)
        x = self.fcs(x)
        return x

    def prepare_data(self, x):
        return x

    def forward_without_classification(self, input):
        x = self.conv(input)
        for fc in self.fcs[:-1]:
            x = fc(x)
        return x

# HT ##############################################################################################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, *args, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply mask if there is one
        if 'mask' in kwargs and kwargs['mask'] is not None:
            mask = kwargs['mask']
            dots.masked_fill_(mask, torch.finfo(torch.float32).min)

        attention = self.attend(dots)
        attention = self.dropout(attention)  # TODO

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attention


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, feedforward_mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, feedforward_mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, *args, **kwargs):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x, *args, **kwargs)
            x = out + x
            x = prenorm_feedforward(x) + x
        return x, attention  # last layer


class PhysioTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.5, emb_dropout=0.5, output='multi'):
        """

        # a token is a time slice of data on a single channel

        @param num_timesteps: int: number of timesteps in each sample
        @param num_channels: int: number of channels of the input data
        @param output: str: can be 'single' or 'multi'. If 'single', the output is a single number to be put with sigmoid activation. If 'multi', the output is a vector of size num_classes to be put with softmax activation.
        note that 'single' only works when the number of classes is 2.
        """
        if output == 'single':
            assert num_classes == 2, 'output can only be single when num_classes is 2'
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.window_duration = window_duration

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.path_embed_dim = patch_embed_dim
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length

        self.grid_dims = self.num_channels, self.num_windows
        self.num_patches = self.num_channels * self.num_windows
        # self.num_patches = 4313
        # self.num_patches = 1343

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
        #     # nn.Conv2d(1, patch_embed_dim, kernel_size=(1, int(self.patch_length*1.5)), stride=(1, self.patch_length), bias=True),
        #     nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
        # )
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
        #     nn.Conv2d(1, patch_embed_dim, (1, 25), (1, 1)),
        #     nn.Conv2d(patch_embed_dim, patch_embed_dim, (22, 1), (1, 1)),
        #     nn.BatchNorm2d(patch_embed_dim),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
        #     nn.Dropout(0.5),
        # )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, 25), stride=(1, 1), bias=True),
            nn.Conv2d(patch_embed_dim, patch_embed_dim, (22, 1), (1, 1)),
            nn.BatchNorm2d(patch_embed_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1343, patch_embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, attn_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, num_classes))
            # self.mlp_head = nn.Sequential(
            #     # nn.Linear(576, 256),
            #     # nn.Linear(12544, 256),
            #     nn.Linear(3904, 256),
            #     nn.ELU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(256, 32),
            #     nn.ELU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(32, 4)
            # )


    def forward(self, x_eeg):
        x = self.encode(x_eeg)
        return self.mlp_head(x)

    def encode(self, x_eeg):
        x = self.to_patch_embedding(x_eeg)
        # _, _, c, t = x.shape
        # _mask = torch.tile(torch.triu(torch.ones(t, t), diagonal=1), (c, c))
        # mask = torch.zeros((c*t+1, c*t+1)).to(x.device)  # cls token can attend to all tokens
        # mask[1:, 1:] = _mask  # add the mask without cls token
        # mask = mask.bool()
        mask = None

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        x, att_matrix = self.transformer(x, mask=mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # x = rearrange(x[:, 1:], 'b n d -> b (n d)')
        x = self.to_latent(x)
        return x

# HCT ###############################################################################################
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        inverse_frequency = 1. / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        inverse_frequency = torch.unsqueeze(inverse_frequency, dim=0)  # unsequeeze for broadcasting to batches
        self.register_buffer('inverse_frequency', inverse_frequency)

    def forward(self, p):
        # t has shape (batch_size, seq_len, embed_dim)
        outer_product = torch.einsum('bn,nd->bnd', p, self.inverse_frequency)   # b=batch size, n=number of tokens,
        pos_emb = torch.cat([outer_product.sin(), outer_product.cos()], dim=-1)
        return pos_emb

class Convalue(nn.Module):
    def __init__(self, conv_channels=8, heads=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(1, 25), stride=(1, 1)),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels * 2, kernel_size=(1, 10), stride=(1, 1))
        )
        self.heads = heads

    def forward(self, x):
        x = rearrange(x, 'b t d -> b 1 t d')
        x_list = []
        for i in range(self.heads):
            x_list.append(self.conv_layers(x))
        x = torch.stack(x_list)
        # x = repeat(x, 'b c t d -> (b h) c t d', h=self.heads)
        # x = self.conv_layers(x)
        x = rearrange(x, 'h b c t d -> b h t (c d)', h=self.heads)
        return x

class ConvalueAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.conv_layers = Convalue(conv_channels=2, heads=heads)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(992, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        v = self.conv_layers(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attention = self.attend(dots)
        attention = self.dropout(attention)  # TODO

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attention

class ConvalueTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, feedforward_mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvalueAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, feedforward_mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x)
            x = out + x
            x = prenorm_feedforward(x) + x
        return x, attention  # last layer

class HierarchicalConvalueTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.5, emb_dropout=0.5, output='multi',
                 pos_embed_mode='learnable'):
    # def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=2, num_heads=5,
    #              feedforward_mlp_dim=64, window_duration=0.1, pool='cls', patch_embed_dim=128, dim_head=64, attn_dropout=0., emb_dropout=0., output='single'):
        """

        # a token is a time slice of data on a single channel

        @param num_timesteps: int: number of timesteps in each sample
        @param num_channels: int: number of channels of the input data
        @param output: str: can be 'single' or 'multi'. If 'single', the output is a single number to be put with sigmoid activation. If 'multi', the output is a vector of size num_classes to be put with softmax activation.
        note that 'single' only works when the number of classes is 2.
        @param pos_embed_mode: str: can be 'learnable' or 'sinusoidal'. If 'learnable', the positional embedding is learned.
        If 'sinusoidal', the positional embedding is sinusoidal. The sinusoidal positional embedding requires meta_info to be passed in with forward
        """
        if output == 'single':
            assert num_classes == 2, 'output can only be single when num_classes is 2'
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.window_duration = window_duration

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.patch_embed_dim = patch_embed_dim
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length

        self.grid_dims = self.num_channels, self.num_windows
        self.num_patches = self.num_channels * self.num_windows

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (nw ps) -> b (c nw) (ps)', nw=self.num_windows, ps=self.patch_length),
        #     nn.Linear(self.patch_length, patch_embed_dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b eegc t -> b 1 eegc t', eegc=self.num_channels, t=self.num_timesteps),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
            # Rearrange('b patchEmbed eegc nPatch -> b patchEmbed (eegc nPatch)', patchEmbed=patch_embed_dim),
        )
        # self.to_patch_embedding = ConvFeatureExtractionModel(self.extraction_layers, dropout=emb_dropout)
        # x = torch.randn(10, self.num_channels, self.num_timesteps)

        self.pos_embed_mode = pos_embed_mode
        self.learnable_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.sinusoidal_pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ConvalueTransformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, attn_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, num_classes))

    def forward(self, x_eeg, *args, **kwargs):
            x = self.encode(x_eeg, *args, **kwargs)
            return self.mlp_head(x)

    def encode(self, x_eeg, *args, **kwargs):
        x = self.to_patch_embedding(x_eeg)
        # x = rearrange(x_eeg, 'b c (t w) -> b (c w) t', t=self.patch_length)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed_mode == 'sinusoidal':
            channel_pos = x['channel_positions'] # batch_size x num_channels
            assert channel_pos.shape[1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
            time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in range(b)])  # batch_size x num_windows  # use sample-relative time positions

            # time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)  # in case where each batach has different time positions
            time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(b, self.num_channels, 1, 1)
            channel_pos_embed = self.sinusoidal_pos_embedding(channel_pos).unsqueeze(2).repeat(1, 1, self.num_windows, 1)
            time_pos_embed = rearrange(time_pos_embed, 'b c t d -> b (c t) d')
            channel_pos_embed = rearrange(channel_pos_embed, 'b c t d -> b (c t) d')

            pos_embed = time_pos_embed + channel_pos_embed
            cls_tokens_pos_embedding = repeat(self.learnable_pos_embedding[:, -1, :], '1 d -> b 1 d', b=b)
            pos_embed = torch.concatenate([pos_embed, cls_tokens_pos_embedding], dim=1)

        elif self.pos_embed_mode == 'learnable':
            pos_embed = self.learnable_pos_embedding[:, :(n + 1)]
        else:
            raise ValueError(f"pos_embed_mode must be either 'sinusoidal' or 'learnable', but got {self.pos_embed_mode}")

        x += pos_embed
        x = self.dropout(x)

        x, att_matrix = self.transformer(x)
        # att_matrix = att_matrix[:, :, 1:, 1:]
        # att_matrix = att_matrix / torch.sum(att_matrix, dim=3, keepdim=True)
        # att_matrix = torch.sum(att_matrix, dim=2)
        # att_matrix = att_matrix / torch.sum(att_matrix, dim=2,keepdim= True)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

    def prepare_data(self, x):
        return x

    def reset(self):
        """
        HT does not have reset defined
        @return:
        """
        pass

# RHT #########################################################################################################
# import warnings
#
# import torch
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# from torch import nn
#
# from renaanalysis.learning.HT import MaskLayer
# from renaanalysis.models.model_utils import init_weight
#
#
# class SinusoidalPositionalEmbedding(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#         inverse_frequency = 1. / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
#         inverse_frequency = torch.unsqueeze(inverse_frequency, dim=0)  # unsequeeze for broadcasting to batches
#         self.register_buffer('inverse_frequency', inverse_frequency)
#
#     def forward(self, p):
#         # t has shape (batch_size, seq_len, embed_dim)
#         outer_product = torch.einsum('bn,nd->bnd', p, self.inverse_frequency)  # b=batch size, n=number of tokens,
#         pos_emb = torch.cat([outer_product.sin(), outer_product.cos()], dim=-1)
#         return pos_emb
#
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class PrePostNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.post_norm = nn.LayerNorm(dim)
#         self.pre_norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.post_norm(self.fn(self.pre_norm(x), **kwargs))
#
#
# class FeedForwardResidualPostNorm(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#         self.lnorm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         return self.lnorm(self.net(x) + x)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class RecurrentGeneralizedPFAttention(nn.Module):
#     def __init__(self, embedding_dim, pos_embed_mode, num_heads=8, dim_head=64, drop_attention=0., dropout=0.1,
#                  pos_embed_activation=None):
#         super().__init__()
#         all_heads_dim = dim_head * num_heads
#         project_out = not (num_heads == 1 and dim_head == embedding_dim)
#
#         self.pos_embed_mod = pos_embed_mode
#         self.dim_head = dim_head
#         self.num_heads = num_heads
#         self.scale = dim_head ** -0.5
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.drop_attention = nn.Dropout(drop_attention)
#
#         self.to_qkv = nn.Linear(embedding_dim, all_heads_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(all_heads_dim, embedding_dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#         self.k_r_time_net = nn.Sequential(
#             nn.Linear(embedding_dim, all_heads_dim, bias=False),
#             # nn.GELU()
#         )
#         self.k_r_channel_net = nn.Sequential(
#             nn.Linear(embedding_dim, all_heads_dim, bias=False),
#             # nn.GELU()
#         )
#         self.k_r_participant_net = nn.Sequential(
#             nn.Linear(embedding_dim, all_heads_dim, bias=False),
#             # nn.GELU()
#         )
#         self.layer_norm = nn.LayerNorm(embedding_dim)
#
#     # def forward(self, x, r_t, r_c, bias_time_e, bias_time_r, bias_channel_r, bias_channel_e):
#     def forward(self, x, r_t, r_c, r_p, bias_pf, mems, qlen):
#         """
#
#         @param x:
#         @param r_t:
#         @param r_c:
#         @param bias_time_e:  num_head x dim_head
#         @param bias_time_r:  num_head x dim_head
#         @param bias_channel_r:  num_head x dim_head
#         @param bias_channel_e:  num_head x dim_head
#         @return:
#         # """
#         b, _, dpatch = x.shape
#
#         if mems is not None:
#             mem_x, mem_r_t, mem_r_c, mem_r_p = mems
#             if b != mem_x.shape[0]:
#                 mem_x, mem_r_t, mem_r_c, mem_r_p = mem_x[:b], mem_r_t[:b], mem_r_c[:b], mem_r_p[:b]
#             # x_with_mems = torch.cat([mem_x, x], dim=1)
#             klen = x.size(1)
#             # x_with_mems = self.layer_norm(torch.cat([mem_x, x], dim=1))
#             r_t = torch.cat([mem_r_t, r_t], dim=1) if r_t.size(1) != klen else r_t
#             r_c = torch.cat([mem_r_c, r_c], dim=1) if r_c.size(1) != klen else r_t
#             r_p = torch.cat([mem_r_p, r_p], dim=1) if r_p.size(1) != klen else r_t
#
#             qkv = self.to_qkv(x).chunk(3, dim=-1)
#             Ex_Wq, Ex_Wke, v = map(lambda t: rearrange(t, 'b n (h d) -> n b h d', h=self.num_heads), qkv)
#             Ex_Wq = Ex_Wq[-qlen:]
#         else:
#             klen = x.size(1)
#             qkv = self.to_qkv(x).chunk(3, dim=-1)
#             # qkv = self.to_qkv(self.layer_norm(x)).chunk(3, dim=-1)
#             Ex_Wq, Ex_Wke, v = map(lambda t: rearrange(t, 'b n (h d) -> n b h d', h=self.num_heads), qkv)
#
#         # Ex_Wq = Ex_Wq.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
#         # Ex_Wke = Ex_Wke.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
#         # v = v.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
#
#         # Ex_Wq = rearrange(Ex_Wq, 'b h n d -> n b h d')
#         # Ex_Wke = rearrange(Ex_Wke, 'b h n d -> n b h d')
#         # v = rearrange(v, 'b h n d -> n b h d')
#
#         # generalized pf attention ########################################################################
#         # repeated_r_p = torch.cat([r_p, repeat(r_p[:, 1:], 'b 1 d-> b k d', k=ntoken - 2)], dim=1)
#         # repeated_W_kr_R_p = rearrange(self.k_r_participant_net(repeated_r_p), 'b n (h d)-> n b h d', h=self.num_heads)  # n = 1 + 1 the first is cls token, the second is the participant token, same for all tokens in a sample
#         # repeated_BD_p = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, repeated_W_kr_R_p))
#
#         Ex_Wq_e_biased = Ex_Wq + bias_pf  # batch_size, n query, num_heads, dim_head treating bias_time_e as the generalized content bias
#
#         # time relative position bias
#         if self.pos_embed_mod == 'sinusoidal':
#             W_kr_R_t = rearrange(self.k_r_time_net(r_t), 'b n (h d)-> n b h d', h=self.num_heads)
#             W_kr_R_c = rearrange(self.k_r_channel_net(r_c), 'b n (h d)-> n b h d', h=self.num_heads)
#             W_kr_R_p = rearrange(self.k_r_participant_net(r_p), 'b n (h d)-> n b h d',
#                                  h=self.num_heads)  # n = 1 + 1 the first is cls token, the second is the participant token, same for all tokens in a sample
#         else:
#             W_kr_R_t = rearrange(r_t, 'b n h d -> n b h d')
#             W_kr_R_c = rearrange(r_c, 'b n h d -> n b h d')
#             W_kr_R_p = rearrange(r_p, 'b n h d -> n b h d')
#
#         # without relative shift ######
#         # Ex_Wke_r_biased = Ex_Wke + (W_kr_R_t + W_kr_R_c) * 0.5   # batch_size, n query, num_heads, dim_head
#         # dots = torch.einsum('ibhd,jbhd->ijbh', Ex_Wq_e_biased, Ex_Wke_r_biased) * self.scale
#
#         # with relative shift ######
#         AC = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, Ex_Wke))  # qlen x klen x bsz x n_head
#         BD_t = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, W_kr_R_t))
#         BD_t = self._rel_shift(BD_t)
#         BD_c = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, W_kr_R_c))
#         BD_c = self._rel_shift(BD_c)
#
#         BD_p = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, W_kr_R_p))
#         # repeat the participant token to match the shape of the other tokens
#         # BD_p_repeats = repeat(BD_p[:, 1:], 'q 1 b h-> q k b h', k=qlen - 2)  # ntoken - 2, one is the cls token, the other is the participant token being repeated
#         # BD_p = torch.cat([BD_p, BD_p_repeats], dim=1)  # ntoken, batch_size, num_heads, dim_head
#         BD_p = self._rel_shift(BD_p)
#
#         BD_ = (BD_t + BD_c + BD_p)
#         BD_.mul_(1 / 3)
#         dots = AC + BD_  # times 0.5 to normalize across multiple positional features
#         dots.mul_(self.scale)  # scale down the dot product
#
#         # transformer-xl attention ########################################################################
#         # r = r_t + r_c
#         # # W_kr_R_t = self.k_r_time_net(r).view(ntoken, b, self.num_heads, self.dim_head)
#         # W_kr_R_t = self.k_r_time_net(r)
#         # W_kr_R_t = rearrange(W_kr_R_t, 'b n (h d)-> n b h d', h=self.num_heads)
#         #
#         # rw_head_q = Ex_Wq + bias_time_e                                         # qlen x bsz x n_head x d_head
#         # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, Ex_Wke))             # qlen x klen x bsz x n_head
#         # rr_head_q = Ex_Wq + bias_time_r
#         # BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, W_kr_R_t))
#         # BD = self._rel_shift(BD)
#         #
#         # dots = AC + BD
#         # dots.mul_(self.scale)
#         ##################################################################################################
#
#         attention = self.softmax(dots)
#         attention = self.drop_attention(attention)  # n query, n query, batch_size, num_heads
#
#         out = torch.torch.einsum('ijbn,jbnd->ibnd', (attention, v))
#         out = rearrange(out, 'n b h d -> b n (h d)')
#
#         '''
#         # vanilla attention
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attention = self.softmax(dots)
#         attention = self.drop_attention(attention)  # TODO
#
#         out = torch.matmul(attention, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         '''
#
#         return self.to_out(out), rearrange(attention, 'q k b h -> b h q k')
#
#     def _rel_shift(self, x, zero_triu=False):
#         zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
#         x_padded = torch.cat([zero_pad, x], dim=1)
#         x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
#         x = x_padded[1:].view_as(x)
#         if zero_triu:
#             ones = torch.ones((x.size(0), x.size(1)))
#             x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
#
#         return x
#
#
# class RecurrentGeneralizedPFTransformer(nn.Module):
#     """
#     self.mems
#      [  [layerouts, r_t, r_c, r_p] ]
#     """
#
#     def __init__(self, embedding_dim, depth, num_heads, dim_head, feedforward_mlp_dim, pos_embed_mode,
#                  drop_attention=0., dropout=0.1, mem_len=0):
#         super().__init__()
#
#         self.depth = depth
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(embedding_dim,
#                         RecurrentGeneralizedPFAttention(embedding_dim, pos_embed_mode, num_heads=num_heads,
#                                                         dim_head=dim_head, drop_attention=drop_attention,
#                                                         dropout=dropout)),
#                 # RecurrentGeneralizedPFAttention(embedding_dim, num_heads=num_heads, dim_head=dim_head, drop_attention=drop_attention, dropout=dropout),
#                 PreNorm(embedding_dim, FeedForward(embedding_dim, feedforward_mlp_dim, dropout=dropout))
#                 # use pre norm for the attention output residual
#             ]))
#         self.num_heads = num_heads
#         self.dim_head = dim_head
#         self.bias_pf = nn.Parameter(torch.randn(self.num_heads, self.dim_head))
#
#         # self.weight_init()
#         # list object to store attention results from past forward passes
#         self.mems = None
#         self.mem_len = mem_len
#         self.num_embeds = 4  # x, r_t, r_c, r_p
#
#     def forward(self, x, r_t, r_c, r_p):
#         """
#
#         @param x:
#         @param r_t: relative time positional embedding
#         @param r_c: relative channel positional embedding
#         @return:
#         """
#         if self.mems is None:
#             self.init_mems()
#         b = x.size(0)
#         qlen = x.size(1)
#         klen = self.mem_len + qlen
#         layer_outs_rs = []
#         if self.mem_len > 0: layer_outs_rs.append((x, r_t[:, -qlen:], r_c, r_p))
#         for i, (attention_layer, prenorm_ff_residual_postnorm) in enumerate(self.layers):
#             if self.mems is not None:
#                 mem_x, _, _, _ = self.mems[i]
#                 if b != mem_x.shape[0]:
#                     mem_x = mem_x[:b]
#                 x_with_mems = torch.cat([mem_x, x], dim=1)  # concate x with mem here so they are prenorm together
#                 out, attention = attention_layer(x_with_mems, r_t=r_t, r_c=r_c, r_p=r_p, bias_pf=self.bias_pf,
#                                                  mems=self.mems[i], qlen=qlen)
#             else:
#                 out, attention = attention_layer(x, r_t=r_t, r_c=r_c, r_p=r_p, bias_pf=self.bias_pf, mems=None,
#                                                  qlen=qlen)
#             # out, attention = prenorm_attention(x, r_t=r_t, r_c=r_c, bias_time_e=self.bias_time_e, bias_time_r=self.bias_time_r, bias_channel_r=self.bias_channel_r, bias_channel_e=self.bias_channel_e)
#             x = out + x  # residual connection
#             x = prenorm_ff_residual_postnorm(x) + x
#
#             if self.mem_len > 0: layer_outs_rs.append((x, r_t[:, -qlen:], r_c, r_p))  # r_t can be of the same len as t
#         self.update_mems(layer_outs_rs, qlen, b)
#         return x, attention  # last layer
#
#     def init_mems(self):
#         """
#         mem will be none if mem_len is 0
#         @return:
#         """
#         if self.mem_len > 0:
#             self.mems = []
#             param = next(self.parameters())
#             for i in range(self.depth + 1):
#                 empty = [torch.empty(0, dtype=param.dtype, device=param.device) for _ in range(self.num_embeds)]
#                 self.mems.append(empty)
#         else:
#             self.mems = None
#
#     def update_mems(self, layer_outs_rs, qlen, b):
#         """
#         There are `mlen + qlen` steps that can be cached into mems
#         For the next step, the last `ext_len` of the `qlen` tokens
#         will be used as the extended context. Hence, we only cache
#         the tokens from `mlen + qlen - self.ext_len - self.mem_len`
#         to `mlen + qlen - self.ext_len`.
#         @param layer_outs_rs:
#         @param qlen:
#         @return:
#         """
#         if self.mems is None: return
#         assert len(layer_outs_rs) == len(self.mems)
#         cur_mem_len = self.mems[0][0].size(1) if self.mems[0][0].numel() != 0 else 0
#         b_mismatch = len(self.mems[0][0]) - b  # check the batch size
#         with torch.no_grad():
#             new_mems = []
#             end_idx = cur_mem_len + max(0, qlen)
#             beg_idx = max(0, end_idx - self.mem_len)
#             for layer_index in range(len(layer_outs_rs)):
#                 # only append mem when 1) mems are empty 2) the number of tokens between the mem and the embeddings matches, when they don't match, it implies the caller is managing the mems for this embedding
#                 if b_mismatch != 0:
#                     layer_outs_rs[layer_index] = [torch.concatenate(
#                         (self.mems[layer_index][embed_index][-b_mismatch:], layer_outs_rs[layer_index][embed_index]),
#                         dim=0)
#                                                   for embed_index in range(self.num_embeds)]
#                 cat = [torch.cat([self.mems[layer_index][embed_index], layer_outs_rs[layer_index][embed_index]], dim=1)
#                        for embed_index in range(self.num_embeds)]
#                 new_mems.append([c[:, beg_idx:end_idx].detach() for c in cat])
#         self.mems = new_mems
#     # def weight_init(self):
#     #     init_weight(self.bias_time_e)
#     #     init_weight(self.bias_time_r)
#     #     init_weight(self.bias_channel_r)
#     #     init_weight(self.bias_channel_e)
#
#
# class RecurrentPositionalFeatureTransformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         raise NotImplementedError
#
#
# class RecurrentHierarchicalTransformer(nn.Module):
#     def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8,
#                  feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
#                  patch_embed_dim=128, dim_head=64, attn_dropout=0.0, emb_dropout=0.1, dropout=0.1, output='multi',
#                  n_participant=13, mem_len=0,
#                  reset_mem_each_session=False, pos_embed_mode='learnable'):
#         """
#
#         # a token is a time slice of data on a single channel
#
#         @param num_timesteps: int: number of timesteps in each sample
#         @param num_channels: int: nusmber of channels of the input data
#         @param output: str: can be 'single' or 'multi'. If 'single', the output is a single number to be put with sigmoid activation. If 'multi', the output is a vector of size num_classes to be put with softmax activation.
#         note that 'single' only works when the number of classes is 2.
#         @param reset_mem_each_session:
#         """
#         if output == 'single':
#             assert num_classes == 2, 'output can only be single when num_classes is 2'
#         super().__init__()
#         self.depth = depth
#         self.num_heads = num_heads
#         self.dim_head = dim_head
#         self.window_duration = window_duration
#
#         self.num_channels = num_channels
#         self.num_timesteps = num_timesteps
#         self.patch_embed_dim = patch_embed_dim
#         self.patch_length = int(window_duration * sampling_rate)
#         self.num_windows = num_timesteps // self.patch_length
#
#         self.grid_dims = self.num_channels, self.num_windows
#         self.num_patches = self.num_channels * self.num_windows
#
#         self.max_tlen = self.num_windows * (mem_len + 1)  # mem plus the current query
#
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
#             nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
#         )
#         self.pos_embed_mode = pos_embed_mode
#         if self.pos_embed_mode == 'sinusoidal':
#             self.pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)
#         elif self.pos_embed_mode == 'learnable':
#             self.learnable_time_embedding = nn.Parameter(torch.randn(1, self.max_tlen, num_heads, dim_head))
#             self.learnable_channel_embedding = nn.Parameter(torch.randn(1, self.num_channels, num_heads, dim_head))
#             self.learnable_participant_embedding_list = nn.Parameter(torch.randn(n_participant, 1, 1, self.num_heads,
#                                                                                  self.dim_head))  # keeps embedding for every participant
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
#         self.cls_token_pos_embedding = nn.Parameter(
#             torch.randn(1, 1, patch_embed_dim)) if pos_embed_mode == 'sinusoidal' else nn.Parameter(
#             torch.randn(1, 1, num_heads, dim_head))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.mem_len = (self.num_channels * self.num_windows * mem_len + 1) if mem_len != 0 else 0  # +1 for cls token
#         self.transformer = RecurrentGeneralizedPFTransformer(patch_embed_dim, depth, num_heads, dim_head,
#                                                              feedforward_mlp_dim, pos_embed_mode,
#                                                              drop_attention=attn_dropout, dropout=dropout,
#                                                              mem_len=self.mem_len)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         if output == 'single':
#             self.mlp_head = nn.Sequential(
#                 nn.LayerNorm(patch_embed_dim),
#                 nn.Linear(patch_embed_dim, 1))
#         else:
#             self.mlp_head = nn.Sequential(
#                 nn.LayerNorm(patch_embed_dim),
#                 nn.Linear(patch_embed_dim, num_classes))
#
#         randomized_participant_pos = torch.randperm(n_participant).to(torch.float32)
#         self.register_buffer('randomized_participant_pos', randomized_participant_pos)
#
#         self.current_session = None
#         self.reset_mem_each_session = reset_mem_each_session
#
#     def forward(self, x, *args, **kwargs):
#         """
#         auditory oddball meta info: 'subject_id', 'run', 'epoch_start_times', 'channel_positions', 'channel_voxel_indices'
#         @param x_eeg:
#         @param meta_info: meta_info is a dictionary
#         @return:
#         """
#
#         # check meta info is complete
#         x = self.encode(x, *args, **kwargs)
#         return self.mlp_head(x)
#
#     def encode(self, x, *args, **kwargs):
#         """
#         currently
#
#         @param x_eeg:
#         @param args:
#         @param kwargs:
#         @return:
#         """
#         x_eeg = x['eeg']
#
#         b, nchannel, _ = x_eeg.shape
#
#         if self.reset_mem_each_session and self.current_session != (current_session := x['session'][0].item()):
#             self.reset()
#             print(f"Current session changed from {self.current_session} to {current_session}. Memory reset.")
#             self.current_session = current_session
#
#         # get discretized time for each token
#         # discretized_start_times = args[3]  // self.window_duration
#         mem_timesteps = int(self.transformer.mems[0][0].size(1) / self.num_channels) if (
#                     self.transformer.mems is not None and torch.numel(self.transformer.mems[0][0]) != 0) else 0
#         mem_num_epochs = mem_timesteps // self.num_windows
#         tlen = mem_timesteps + self.num_windows
#
#         participant_pos = torch.unique(x['subject_id']).item()
#
#         if self.pos_embed_mode == 'sinusoidal':
#             # time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
#             # time_pos = torch.stack([torch.arange(a, a+self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use session-relative time positions
#             # time_pos = torch.stack([torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
#             time_pos = torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long)[None,
#                        :]  # batch_size x num_windows  # use sample-relative time positions
#
#             # compute channel positions that are voxel discretized
#             channel_pos = x['channel_voxel_indices']  # batch_size x num_channels
#
#             # get the participant embeddings
#             participant_pos = self.randomized_participant_pos[participant_pos][:,
#                               None]  # batch_size x 1, get random participant pos
#
#             # each sample in a batch must have the same participant embedding
#             time_pos_embed = self.pos_embedding(time_pos)
#             channel_pos_embed = self.pos_embedding(channel_pos)
#             participant_pos_embed = self.pos_embedding(
#                 participant_pos)  # no need to repeat because every token in the same sample has the same participant embedding
#
#             time_pos_embed = time_pos_embed.unsqueeze(1).repeat(b, nchannel, 1, 1).reshape(b, -1, self.patch_embed_dim)
#             channel_pos_embed = channel_pos_embed.unsqueeze(2).repeat(1, 1, self.num_windows, 1).reshape(b, -1,
#                                                                                                          self.patch_embed_dim)
#             participant_pos_embed = repeat(participant_pos_embed, 'b 1 d-> b n d', n=self.num_patches)
#         else:  # learnable
#             time_pos_embed = self.learnable_time_embedding[:, -tlen:]
#             channel_pos_embed = self.learnable_channel_embedding
#             participant_pos_embed = repeat(self.learnable_participant_embedding_list[int(participant_pos)],
#                                            '1 1 h d -> b n h d', b=b, n=self.num_patches)  # repeat for batch
#
#             time_pos_embed = repeat(time_pos_embed.unsqueeze(1), '1 1 t h d -> b (c t) h d', b=b, c=nchannel)
#             channel_pos_embed = repeat(channel_pos_embed.unsqueeze(2), '1 c 1 h d -> b (c t) h d', b=b,
#                                        t=self.num_windows)
#         # viz_time_positional_embedding(time_pos_embed)  # time embedding differs in batch
#         # viz_time_positional_embedding(channel_pos_embed)  # time embedding differs in batch
#
#         # prepare the positional features that are different among tokens in the same sample
#
#         x = self.to_patch_embedding(x_eeg)
#         x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#
#         b, ntoken, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         if self.pos_embed_mode == 'sinusoidal':
#             cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 d -> b 1 d', b=b)
#         else:
#             cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 h d -> b 1 h d', b=b)
#
#         # insert cls pos embedding based on the number of mem steps
#         for i in range(mem_num_epochs + 1):
#             time_pos_embed = torch.cat(
#                 (time_pos_embed[:, :i * ntoken, :], cls_tokens_pos_embedding, time_pos_embed[:, i * ntoken:, :]),
#                 dim=1)  # only time pos embedding can be of klen before the transformer
#         channel_pos_embed = torch.cat((cls_tokens_pos_embedding, channel_pos_embed), dim=1)
#         participant_pos_embed = torch.cat((cls_tokens_pos_embedding, participant_pos_embed), dim=1)
#
#         x = self.dropout(x)
#
#         x, att_matrix = self.transformer(x, time_pos_embed, channel_pos_embed, participant_pos_embed)
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#         x = self.to_latent(x)
#         return x
#
#     def reset(self):
#         # warnings.warn("RHT.reset(): To be implemented")
#         self.transformer.init_mems()


batch_size = 8
n_epochs = 500
c_dim = 4
lr = 0.0002
# lr = 1e-5
b1 = 0.5
b2 = 0.999
dimension = (190, 50)


criterion_l1 = torch.nn.L1Loss().cuda()
criterion_l2 = torch.nn.MSELoss().cuda()
criterion_cls = torch.nn.CrossEntropyLoss().cuda()

train_data = train_epochs.get_data().copy()
train_data = np.expand_dims(train_data, axis=1)

train_label = train_epochs.events[:, -1]
# shuffle_num = np.random.permutation(len(train_data))

# train_data = train_data[shuffle_num, :, :, :]
# train_label = train_label[shuffle_num]

test_data = eval_epochs.get_data().copy()
test_data = np.expand_dims(test_data, axis=1)
test_label = eval_epochs.events[:, -1]

target_mean = np.mean(train_data)
target_std = np.std(train_data)
train_data = (train_data - target_mean) / target_std
test_data = (test_data - target_mean) / target_std

train_data = torch.from_numpy(train_data)
train_label = torch.from_numpy(train_label)
dataset = TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

test_data = torch.from_numpy(test_data).float().cuda()
test_label = torch.from_numpy(test_label).long().cuda()
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = Conformer().cuda()
# model = model.cuda()
_, _, num_channels, num_timesteps = train_data.shape
# model = EEGCNN(train_data.squeeze().shape, num_classes=4)
# model = PhysioTransformer(num_timesteps=num_timesteps, num_channels=num_channels, sampling_rate=srate, num_classes=4,
#                           window_duration=0.4, feedforward_mlp_dim=128, patch_embed_dim=64)
# model = HierarchicalConvalueTransformer(num_timesteps=num_timesteps, num_channels=num_channels, sampling_rate=srate, num_classes=4,
#                           window_duration=0.5, feedforward_mlp_dim=128, patch_embed_dim=64)
# model = RecurrentHierarchicalTransformer(num_timesteps=num_timesteps, num_channels=num_channels, sampling_rate=srate,
#                                          num_classes=4, window_duration=0.5, pos_embed_mode='learnable')
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

def interaug(timg, label):
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]

        tmp_aug_data = np.zeros((int(batch_size / 4), 1, 22, 1000))
        for ri in range(int(batch_size / 4)):
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                  rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).cuda()
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).cuda()
    aug_label = aug_label.long()
    return aug_data, aug_label

bestAcc = 0
averAcc = 0
num = 0
Y_true = 0
Y_pred = 0

# Train the cnn model
total_step = len(train_dataloader)
curr_lr = lr

for e in range(n_epochs):
    # in_epoch = time.time()
    model.train()
    pred_train_all = torch.Tensor([]).cuda()
    label_train_all = torch.Tensor([]).cuda()
    for i, (img, label) in enumerate(train_dataloader):

        img = img.type(torch.cuda.FloatTensor).cuda()
        label = label.type(torch.cuda.LongTensor).cuda()

        # img = img.cuda().type(torch.cuda.FloatTensor)
        # label = label.cuda().type(torch.cuda.LongTensor)

        # data augmentation
        aug_data, aug_label = interaug(train_data, train_label)
        img = torch.cat((img, aug_data))
        label = torch.cat((label, aug_label))

        # if isinstance(model, RecurrentHierarchicalTransformer):
        #     img = torch.squeeze(img)
        #     outputs = model({'eeg': img, 'subject_id': torch.Tensor([3] * len(img)).cuda()})
        if isinstance(model, PhysioTransformer) or isinstance(model, HierarchicalConvalueTransformer) or isinstance(model, EEGCNN):
            img = torch.squeeze(img)
            outputs = model(img)
        else:
            tok, outputs = model(img)
        loss = criterion_cls(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_train_all = torch.concat((pred_train_all, outputs))
        label_train_all = torch.concat((label_train_all, label))

    # test process
    if (e + 1) % 1 == 0:
        model.eval()

        # if isinstance(model, RecurrentHierarchicalTransformer):
        #     test_data = torch.squeeze(test_data)
        #     Cls = model({'eeg': test_data, 'subject_id': torch.Tensor([3] * len(test_data)).cuda()})
        y_pred_all = torch.Tensor([]).cuda()
        test_label_all = torch.Tensor([]).cuda()
        for test_data, test_label in test_dataloader:
            if isinstance(model, PhysioTransformer) or isinstance(model, HierarchicalConvalueTransformer) or isinstance(model, EEGCNN):
                test_data = torch.squeeze(test_data)
                Cls = model(test_data)
            else:
                Tok, Cls = model(test_data)

            loss_test = criterion_cls(Cls, test_label)
            y_pred = torch.max(Cls, 1)[1]
            y_pred_all = torch.concat((y_pred_all, y_pred))
            test_label_all = torch.concat((test_label_all, test_label))
        acc = float((y_pred_all == test_label_all).cpu().numpy().astype(int).sum()) / float(test_label_all.size(0))

        train_pred = torch.max(pred_train_all, 1)[1]
        train_acc = float((train_pred == label_train_all).cpu().numpy().astype(int).sum()) / float(label_train_all.size(0))

        print('Epoch:', e,
              '  Train loss: %.6f' % loss.detach().cpu().numpy(),
              '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
              '  Train accuracy %.6f' % train_acc,
              '  Test accuracy is %.6f' % acc)

        num = num + 1
        averAcc = averAcc + acc
        if acc > bestAcc:
            bestAcc = acc
            Y_true = test_label
            Y_pred = y_pred

averAcc = averAcc / num
print('The average accuracy is:', averAcc)
print('The best accuracy is:', bestAcc)

device ='cuda:0' if torch.cuda.is_available() else 'cpu'
if not isinstance(model, Conformer):
    test_dataloader.dataset.tensors = (torch.squeeze(test_dataloader.dataset.tensors[0]), test_dataloader.dataset.tensors[1])
ht_eeg_viz_dataloader_batch(model, test_dataloader, Attention, device, 'BCICompetitionResults',
                            note='', load_saved_rollout=False, head_fusion='mean',
                            cls_colors=event_viz_colors,
                            discard_ratio=0.9, is_pca_ica=False, pca=None, ica=None, batch_size=256, srate=srate,
                            event_ids={name: e_id-1 for name, e_id in event_id.items()}, eeg_channels=list(eeg_channels), picks=('C3', 'C4'))
class_names = list(event_viz_colors.keys())
viz_confusion_matrix(test_label_all.detach().cpu().numpy(), y_pred_all.detach().cpu().numpy(), class_names, n_epochs, 0, 'Test')
viz_confusion_matrix(label_train_all.detach().cpu().numpy(), train_pred.detach().cpu().numpy(), class_names, n_epochs, 0, 'Train')

