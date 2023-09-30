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

from renaanalysis.params.params import verbose

from renaanalysis.params.params import eeg_name


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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

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

    def forward(self, x):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x)
            x = out + x
            x = prenorm_feedforward(x) + x
        return x, attention  # last layer
