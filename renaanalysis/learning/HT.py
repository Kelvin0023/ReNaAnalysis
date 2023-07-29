import copy
import os
import pickle

import torch
from torch import nn

from einops import rearrange, repeat, einops
from einops.layers.torch import Rearrange
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


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

class Convalue(nn.Module):
    def __init__(self, conv_channels=8, heads=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(1, 10), stride=(1, 3)),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels * 2, kernel_size=(1, 10), stride=(1, 2))
        )
        self.heads = heads

    def forward(self, x):
        x = rearrange(x, 'b t d -> b 1 t d')
        x = repeat(x, 'b c t d -> (b h) c t d', h=self.heads)
        x = self.conv_layers(x)
        x = rearrange(x, '(b h) c t d -> b h t (c d)', h=self.heads)
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
            nn.Linear(512, dim),
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

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
        dropout: float = 0.0
    ):
        super().__init__()

        def block(n_in, n_out, kernel_size, stride):
            def make_conv():
                conv = nn.Conv2d(n_in, n_out, kernel_size, stride)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.LayerNorm(dim, elementwise_affine=True),
                nn.GELU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride
                )
            )
            in_d = dim

    def forward(self, x):
        # BxCxT
        # x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        return x

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return torch.tensor(inds)

class MaskLayer(nn.Module):
    def __init__(self, p_t, p_c, c_span, mask_t_span, mask_c_span, t_mask_replacement, c_mask_replacement):
        super(MaskLayer, self).__init__()
        self.p_t = p_t
        self.p_c = p_c
        self.c_span = c_span
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.t_mask_replacement = t_mask_replacement
        self.c_mask_replacement = c_mask_replacement

    def make_t_mask(self, shape, p, span, allow_no_inds=False):
        mask = torch.zeros(shape, requires_grad=False, dtype=bool)

        for i in range(shape[0]):
            mask_seeds = list()
            while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
                mask_seeds = torch.nonzero(torch.rand(shape[1]) < p)

            mask[i, _make_span_from_seeds(mask_seeds, span, total=shape[1])] = True

        return mask

    def make_c_mask(self, shape, p, span, allow_no_inds=False):
        mask = torch.zeros(shape, requires_grad=False, dtype=bool)
        for i in range(shape[0]):
            mask_seeds = list()
            while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
                mask_seeds = torch.nonzero(torch.rand(shape[1]) < p)
            if self.c_span:
                mask[i, _make_span_from_seeds(mask_seeds, span, total=shape[1])] = True
            else:
                mask[i, mask_seeds] = True

        return mask

    def forward(self, x):
        original_input = torch.clone(x)
        (batch_size, patch_dim, Height, Width) = x.shape
        x = x.permute(0, 2, 3, 1)
        # masks = []
        # for i in range(x.shape[0]):
        #     while True:
        #         mask = torch.rand_like(torch.empty(x.shape[2],x.shape[3])) < self.mask_ratio
        #         if torch.any(mask):
        #             break
        #     masks.append(mask.unsqueeze(0).expand(patch_dim, -1, -1))
        # masks = torch.stack(masks).to(x.device)
        # masked_input = x.masked_fill(mask_t, 0.0)
        if self.p_t > 0:
            mask_t = self.make_t_mask((batch_size, Width), self.p_t, self.mask_t_span)
        if self.p_c > 0:
            mask_c = self.make_c_mask((batch_size, Height), self.p_c, self.mask_c_span)
        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.t_mask_replacement
        if mask_c is not None:
            x[mask_c] = self.c_mask_replacement
        x = x.permute(0, 3, 1, 2)
        return x, original_input, mask_t, mask_c


class HierarchicalTransformer(nn.Module):
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

    def forward(self, x_eeg, *args, **kwargs):
            x = self.encode(x_eeg, *args, **kwargs)
            return self.mlp_head(x)

    def encode(self, x_eeg, *args, **kwargs):
        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed_mode == 'sinusoidal':
            channel_pos = args[4]  # batch_size x num_channels
            assert channel_pos.shape[1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
            time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in range(b)])  # batch_size x num_windows  # use sample-relative time positions

            time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)
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
    # def test(self, img):
    #     x = self.to_patch_embedding(img)
    #     b, n, _ = x.shape
    #
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.learnable_pos_embedding[:, :(n + 1)]
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
            channel_pos = args[4]  # batch_size x num_channels
            assert channel_pos.shape[1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
            time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in range(b)])  # batch_size x num_windows  # use sample-relative time positions

            time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)
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

class HierarchicalTransformerContrastivePretrain(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=32, dim_head=32, attn_dropout=0.5, emb_dropout=0.5, output='multi', p_t=0.1, p_c=0.2, mask_t_span=2, mask_c_span=5):
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
        self.HierarchicalTransformer = HierarchicalTransformer(num_timesteps, num_channels, sampling_rate, num_classes, depth=depth, num_heads=num_heads, feedforward_mlp_dim=feedforward_mlp_dim, window_duration=window_duration, pool=pool,
                 patch_embed_dim=patch_embed_dim, dim_head=dim_head, attn_dropout=attn_dropout, emb_dropout=emb_dropout, output=output)
        self.mask_layer = MaskLayer(p_t=p_t, p_c=p_c, c_span=False, mask_t_span=mask_t_span, mask_c_span=mask_c_span,
                                    t_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_channels, self.path_embed_dim), requires_grad=True),
                                    c_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_windows, self.path_embed_dim), requires_grad=True))
    def forward(self, x_eeg):
        x = self.HierarchicalTransformer.to_patch_embedding(x_eeg)
        x, original_x, mask_t, mask_c = self.mask_layer(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.HierarchicalTransformer.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.HierarchicalTransformer.learnable_pos_embedding[:, :(n + 1)]
        x = self.HierarchicalTransformer.dropout(x)

        x, att_matrix = self.HierarchicalTransformer.transformer(x)
        # att_matrix = att_matrix[:, :, 1:, 1:]
        # att_matrix = att_matrix / torch.sum(att_matrix, dim=3, keepdim=True)
        # att_matrix = torch.sum(att_matrix, dim=2)
        # att_matrix = att_matrix / torch.sum(att_matrix, dim=2,keepdim= True)
        x = self.HierarchicalTransformer.to_latent(x[:, 1:].transpose(1, 2).view(original_x.shape))  # exclude cls
        return x, original_x, mask_t, mask_c

    def prepare_data(self, x):
        return x


def viz_ht(model: HierarchicalTransformer, x_eeg, y, label_encoder):
    pass
    model.eval()

    torch.save(model.state_dict(), os.path.join('HT/RSVP-itemonset-locked', 'model.pt'))
    pickle.dump(x_eeg, open(os.path.join('HT/RSVP-itemonset-locked', 'x_eeg.pkl'), 'wb'))
    pickle.dump(y, open(os.path.join('HT/RSVP-itemonset-locked', 'y.pkl'), 'wb'))
    pickle.dump(label_encoder, open(os.path.join('HT/RSVP-itemonset-locked', 'label_encoder.pkl'), 'wb'))

class SimularityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_tokens, original_tokens):
        return torch.mean(torch.sum(-F.cosine_similarity(pred_tokens.permute(0, 2, 3, 1), original_tokens.permute(0, 2, 3, 1), dim=-1), dim=(1, 2)))

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, n_neg):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.n_neg = n_neg
        self.loss_fn = nn.CrossEntropyLoss()

    def _generate_negatives(self, orig_tokens):
        """Generate negative samples to compare each sequence location against"""
        batch_size, patch_dim, Height, Width = orig_tokens.shape
        z_k = orig_tokens.permute([0, 3, 2, 1]).reshape(-1, patch_dim)
        full_len = Height * Width
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(0, full_len - 1, size=(batch_size, full_len * self.n_neg))
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size):
                negative_inds[i] += i * full_len

        z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.n_neg, patch_dim)
        return z_k

    def _calculate_similarity(self, unmasked_tokens, contextual_output, negatives):
        contextual_output = contextual_output.unsqueeze(-2)
        unmasked_tokens = unmasked_tokens.unsqueeze(-2)

        negative_in_target = (contextual_output == negatives).all(-1)
        targets = torch.cat([contextual_output, negatives], dim=-2)

        logits = F.cosine_similarity(unmasked_tokens, targets, dim=-1) / self.temperature
        if negative_in_target.any():
            logits[:, :, 1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def _calculate_distance(self, unmasked_tokens, contextual_output, negatives):
        contextual_output = contextual_output.unsqueeze(-2)
        unmasked_tokens = unmasked_tokens.unsqueeze(-2)

        negative_in_target = (contextual_output == negatives).all(-1)
        targets = torch.cat([contextual_output, negatives], dim=-2)

        num_logits = targets.shape[-2]
        logits = rearrange(-torch.norm(unmasked_tokens - targets, dim=-1) / self.temperature, 'b t l -> b (t l)')
        min = logits.min(dim=-1, keepdim=True)[0]
        max = logits.max(dim=-1, keepdim=True)[0]
        logits = rearrange(2 / self.temperature * ((logits - min) / (max - min)) - 1 / self.temperature, 'b (t l) -> b t l', l=num_logits)
        if negative_in_target.any():
            logits[:, :, 1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, pred_tokens, original_tokens, metric='simularity'):
        batch_size, token_dim, num_channels, num_windows = pred_tokens.shape
        negt_tokens = self._generate_negatives(pred_tokens)
        pred_tokens = pred_tokens.permute(0, 2, 3, 1).view(batch_size, -1, token_dim)  # Shape: (32, 576, 128)
        original_tokens = original_tokens.permute(0, 2, 3, 1).view(batch_size, -1, token_dim)  # Shape: (32, 576, 128)
        if metric == 'simularity':
            logits = self._calculate_similarity(original_tokens, pred_tokens, negt_tokens)
        elif metric == 'distance':
            logits = self._calculate_distance(original_tokens, pred_tokens, negt_tokens)
        elif metric == 'both':
            logits = self._calculate_similarity(original_tokens, pred_tokens, negt_tokens) + self._calculate_distance(original_tokens, pred_tokens, negt_tokens)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, labels)
