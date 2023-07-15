import copy
import os
import pickle
import warnings

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from renaanalysis.ht_visualization.visualization import viz_time_positional_embedding
from renaanalysis.models.model_utils import init_weight


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class RecurrentAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, dim_head=64, drop_attention=0., dropout=0.1):
        super().__init__()
        all_heads_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == embedding_dim)

        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.drop_attention = nn.Dropout(drop_attention)

        self.to_qkv = nn.Linear(embedding_dim, all_heads_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(all_heads_dim, embedding_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.k_r_time_net = nn.Linear(embedding_dim, all_heads_dim, bias=False)
        self.k_r_channel_net = nn.Linear(embedding_dim, all_heads_dim, bias=False)

    def forward(self, x, r_t, r_c, bias_time_e, bias_time_r, bias_channel_r, bias_channel_e):
        """

        @param x:
        @param r_t:
        @param r_c:
        @param bias_time_e:  num_head x dim_head
        @param bias_time_r:  num_head x dim_head
        @param bias_channel_r:  num_head x dim_head
        @param bias_channel_e:  num_head x dim_head
        @return:
        """
        b, ntoken, dpatch = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        Ex_Wq, Ek_Wke, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        Ex_Wq = Ex_Wq.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
        Ek_Wke = Ek_Wke.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
        v = v.contiguous().view(ntoken, b, self.num_heads, self.dim_head)

        Ex_Wq_e_biased = Ex_Wq + bias_time_e + bias_channel_e  # batch_size, n query, num_heads, dim_head

        A = torch.einsum('ibhd,jbhd->ijbh', Ex_Wq_e_biased, Ek_Wke)  # n query, n query, batch_size, num_heads

        W_kr_R_t = self.k_r_time_net(r_t).view(ntoken, b, self.num_heads, self.dim_head)
        W_kr_R_c = self.k_r_channel_net(r_c).view(ntoken, b, self.num_heads, self.dim_head)

        Ex_Wq_r_biased = Ex_Wq + bias_time_r + bias_channel_r  # batch_size, n query, num_heads, dim_head

        B = torch.einsum('ibhd,jbhd->ijbh', Ex_Wq_r_biased, W_kr_R_t)  # n query, n query, batch_size, num_heads

        C = torch.einsum('ibhd,jbhd->ijbh', Ex_Wq_r_biased, W_kr_R_c)  # n query, n query, batch_size, num_heads

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = (A + B + C) * self.scale

        attention = self.attend(dots)
        attention = self.drop_attention(attention)  # n query, n query, batch_size, num_heads

        out = torch.torch.einsum('ijbn,jbnd->ibnd', (attention, v))
        out = rearrange(out, 'n b h d -> b n (h d)')
        return self.to_out(out), attention


class RecurrentSpatialTemporalTransformer(nn.Module):
    """

    """
    def __init__(self, dim, depth, num_heads, dim_head, feedforward_mlp_dim, drop_attention=0., dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, RecurrentAttention(dim, num_heads=num_heads, dim_head=dim_head, drop_attention=drop_attention, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, feedforward_mlp_dim, dropout=dropout))
            ]))
        self.num_heads = num_heads
        self.dim_head = dim_head
        # bias terms representing the absolute positional embedding of <positional feature> times the query weights
        # these bias terms are shared across all layers
        self.bias_time_e = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.bias_time_r = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.bias_channel_r = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.bias_channel_e = nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.weight_init()
        # list object to store attention results from past forward passes
        self.memories = None

    def forward(self, x, r_t, r_c):
        """

        @param x:
        @param r_t: relative time positional embedding
        @param r_c: relative channel positional embedding
        @return:
        """
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x, r_t=r_t, r_c=r_c, bias_time_e=self.bias_time_e, bias_time_r=self.bias_time_r, bias_channel_r=self.bias_channel_r, bias_channel_e=self.bias_channel_e)
            x = out + x
            x = prenorm_feedforward(x) + x
        return x, attention  # last layer

    def weight_init(self):
        init_weight(self.bias_time_e)
        init_weight(self.bias_time_r)
        init_weight(self.bias_channel_r)
        init_weight(self.bias_channel_e)

class RecurrentPositionalFeatureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class RecurrentHierarchicalTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.0, emb_dropout=0.1, dropout=0.1, output='multi'):
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
        self.patch_embed_dim = patch_embed_dim
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length

        self.grid_dims = self.num_channels, self.num_windows
        self.num_patches = self.num_channels * self.num_windows

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.cls_token_pos_embedding = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = RecurrentSpatialTemporalTransformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, drop_attention=attn_dropout, dropout=dropout)

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


    def forward(self, x_eeg, meta_info):
        """

        @param x_eeg:
        @param meta_info: meta_info is a dictionary
        @return:
        """
        # check meta info is complete
        x = self.encode(x_eeg, meta_info)
        return self.mlp_head(x)

    def encode(self, x_eeg, meta_info):
        assert 'epoch_start_times' in meta_info and 'epoch_start_times' in meta_info, 'meta_info must contain sampling_rate'
        b, nchannel, _ = x_eeg.shape
        # get discretized time for each token
        discretized_start_times = meta_info['epoch_start_times'] // self.window_duration
        time_pos = torch.stack([torch.arange(a, a+self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows
        # time_pos = time_pos.unsqueeze(1).repeat(1, x_eeg.shape[1], 1).reshape(len(x_eeg), -1)  # batch_size x num_tokens
        # compute channel position magitudes discretized
        channel_pos = meta_info['channel_voxel_indices']  # batch_size x num_channels

        time_pos_embed = self.pos_embedding(time_pos)
        channel_pos_embed = self.pos_embedding(channel_pos)

        # viz_time_positional_embedding(time_pos_embed)  # time embedding differs in batch
        # viz_time_positional_embedding(channel_pos_embed)  # time embedding differs in batch
        time_pos_embed = time_pos_embed.unsqueeze(1).repeat(1, nchannel, 1, 1).reshape(b, -1, self.patch_embed_dim)
        channel_pos_embed = channel_pos_embed.unsqueeze(2).repeat(1, 1, self.num_windows, 1).reshape(b, -1, self.patch_embed_dim)

        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, ntoken, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 d -> b 1 d', b=b)
        time_pos_embed = torch.cat((cls_tokens_pos_embedding, time_pos_embed), dim=1)
        channel_pos_embed = torch.cat((cls_tokens_pos_embedding, channel_pos_embed), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)
        time_pos_embed = self.dropout(time_pos_embed)
        channel_pos_embed = self.dropout(channel_pos_embed)

        x, att_matrix = self.transformer(x, time_pos_embed, channel_pos_embed)
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
        warnings.warn("RHT.reset(): To be implemented")