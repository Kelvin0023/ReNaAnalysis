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
    def __init__(self, depth=1, num_heads=2, feedforward_mlp_dim=12, token_embed_dim=6, dim_head=6, attn_dropout=0.5, emb_dropout=0.5, *args, **kwargs):

        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.token_embed_dim = token_embed_dim

        self.channel_weights_transformer = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, token_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
            # Rearrange('b patchEmbed eegc nPatch -> b patchEmbed (eegc nPatch)', patchEmbed=patch_embed_dim),
            Transformer(token_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, attn_dropout),
            Transformer(token_embed_dim, 1, num_heads, 1, feedforward_mlp_dim, attn_dropout),
        )
        self.to_latent = nn.Identity()


    def forward(self, x, *args, **kwargs):
        assert 'channel_positions' in kwargs and kwargs['channel_positions'] is not None , "ChannelAgnosticTransformer: channel_positions must be passed in."
        channel_positions = kwargs['channel_positions']
        conv_weights = self.channel_weights_transformer(channel_positions)