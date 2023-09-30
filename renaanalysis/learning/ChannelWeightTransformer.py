import copy
import math
import os
import pickle

import torch
from torch import nn

from einops import rearrange, repeat, einops
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import matplotlib.pyplot as plt

from renaanalysis.learning.Transformer import Transformer
from renaanalysis.params.params import verbose

from renaanalysis.params.params import eeg_name


class ChannelAgnosticTransformer(nn.Module):
    def __init__(self, token_embed_dim, depth=1, num_heads=2, feedforward_mlp_dim_factor=2, dim_head_factor=1., dim_head=6, attn_dropout=0.5, emb_dropout=0.5, *args, **kwargs):

        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.token_embed_dim = token_embed_dim

        self.conv_weight_dim  = int(token_embed_dim ** 2)  # the dimension of the convolutional weights, in_channels times out_channels
        self.dim_head = int(dim_head_factor * self.conv_weight_dim)
        self.feedforward_mlp_dim = int(feedforward_mlp_dim_factor * self.conv_weight_dim)

        self.channel_weights_transformer = nn.Sequential(
            Rearrange('b c p -> b 1 c p'),  # p is 3: x, y, z spatial coordinates
            nn.Conv2d(1, self.conv_weight_dim, kernel_size=(1, 3), stride=(1, 1), bias=True),
            Rearrange('b d c 1 -> b c d'),  # c is effective number of tokens here
            Transformer(self.conv_weight_dim, depth, num_heads, dim_head=self.dim_head, feedforward_mlp_dim=self.feedforward_mlp_dim, dropout=attn_dropout, return_attention=False),
            Rearrange('b c (td1 td2) -> b td1 td2 c 1', td1=token_embed_dim, td2=token_embed_dim)
        )


    def forward(self, x, *args, **kwargs):
        b, d, c, t = x.shape
        assert 'channel_positions' in kwargs and kwargs['channel_positions'] is not None , "ChannelAgnosticTransformer: channel_positions must be passed in."
        # alawys take the first channel positions in a batch , a batch should have the same channel positions
        channel_positions = kwargs['channel_positions']
        channel_positions = channel_positions[0:1]
        # TODO assert channel positions are shared between different samples in a batch, before taking the first
        conv_weights = self.channel_weights_transformer(channel_positions)[0]  # take the first from the batch
        conv2d = nn.Conv2d(d, d, (c, 1), (1, 1), bias=False).to(x.device)
        conv2d.weight = nn.Parameter(conv_weights)
        return conv2d(x)
