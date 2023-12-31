import os
import pickle

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

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


# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape
#
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)
#
#         x = self.transformer(x)
#
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#         return self.mlp_head(x)

class HierarchicalTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=32, dim_head=32, attn_dropout=0.5, emb_dropout=0.5, output='multi'):
    # def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=2, num_heads=5,
    #              feedforward_mlp_dim=64, window_duration=0.1, pool='cls', patch_embed_dim=128, dim_head=64, attn_dropout=0., emb_dropout=0., output='single'):
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
        # x = torch.randn(10, self.num_channels, self.num_timesteps)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
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

    def forward(self, x_eeg):
        x = self.encode(x_eeg)
        return self.mlp_head(x)

    def encode(self, x_eeg):
        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
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
    # def test(self, img):
    #     x = self.to_patch_embedding(img)
    #     b, n, _ = x.shape
    #
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x)
    #
    #     x, att_matrix = self.transformer(x)
    #     att_matrix = att_matrix[:, :, 1:, 1:]
    #     att_matrix = att_matrix / torch.sum(att_matrix, dim=3, keepdim=True)
    #     att_matrix = torch.sum(att_matrix, dim=2)
    #     # att_matrix = att_matrix / torch.sum(att_matrix, dim=2,keepdim= True)
    #     # note test does not have lstm and sequence
    #     x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
    #
    #     return self.mlp_head(x), att_matrix

    # def get_grid_size(self):
    #     return self.grid_size

def viz_ht(model: HierarchicalTransformer, x_eeg, y, label_encoder):
    pass
    model.eval()

    torch.save(model.state_dict(), os.path.join('HT/RSVP-itemonset-locked', 'model.pt'))
    pickle.dump(x_eeg, open(os.path.join('HT/RSVP-itemonset-locked', 'x_eeg.pkl'), 'wb'))
    pickle.dump(y, open(os.path.join('HT/RSVP-itemonset-locked', 'y.pkl'), 'wb'))
    pickle.dump(label_encoder, open(os.path.join('HT/RSVP-itemonset-locked', 'label_encoder.pkl'), 'wb'))

