import torch
from torch import nn

from einops import rearrange, repeat, einops
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt

from renaanalysis.learning.HT import PreNorm, Attention, SinusoidalPositionalEmbedding, MaskLayer
from renaanalysis.params.params import verbose


class FeedForwardTrans(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class AutoTransEncoder(nn.Module):
    def __init__(self, depth, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for encoder_layer in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim, dropout=dropout)),
                PreNorm(dim, FeedForwardTrans(dim, 2*dim, dropout=dropout))
            ]))
            dim = 2*dim

    def forward(self, x):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x)
            x = out + x
            x = prenorm_feedforward(x)
        return x, attention  # last layer

class AutoTransDecoder(nn.Module):
    def __init__(self, depth, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for decoder_layer in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim, dropout=dropout)),
                PreNorm(dim, FeedForwardTrans(dim, dim//2, dropout=dropout))
            ]))
            dim = dim//2

    def forward(self, x):
        for prenorm_attention, prenorm_feedforward in self.layers:
            out, attention = prenorm_attention(x)
            x = out + x
            x = prenorm_feedforward(x)
        return x, attention  # last layer

class HierarchicalAutoTranscoder(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.5, emb_dropout=0.5, output='multi',
                 pos_embed_mode='learnable'):
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

        self.num_patches = self.num_channels * self.num_windows
        self.output = output

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (nw ps) -> b (c nw) (ps)', nw=self.num_windows, ps=self.patch_length),
        #     nn.Linear(self.patch_length, patch_embed_dim),
        # )
        # self.to_patch_embedding = nn.Linear(self.patch_length, patch_embed_dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
            # Rearrange('b patchEmbed eegc nPatch -> b patchEmbed (eegc nPatch)', patchEmbed=patch_embed_dim),
        )
        self.to_time_series = nn.Linear(patch_embed_dim, self.patch_length)

        self.pos_embed_mode = pos_embed_mode
        self.learnable_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.sinusoidal_pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = AutoTransEncoder(depth, patch_embed_dim, num_heads, dim_head, attn_dropout)
        self.decoder = AutoTransDecoder(depth, patch_embed_dim * (2 ** depth), num_heads, dim_head, attn_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim*(2**depth)),
                nn.Linear(patch_embed_dim*(2**depth), 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim * (2 ** depth)),
                nn.Linear(patch_embed_dim * (2 ** depth), num_classes))

    def disable_pretrain_parameters(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        if verbose is not None:
            print('decoder gradient disabled')
        for param in self.to_time_series.parameters():
            param.requires_grad = False
        if verbose is not None:
            print('to_time_series gradient disabled')

    def disable_classification_parameters(self):

        for param in self.mlp_head.parameters():
            param.requires_grad = False
        if verbose is not None:
            print('mlp_head parameters disabled')
        self.cls_token.requires_grad = False
        if verbose is not None:
            print('cls_token parameters disabled')
        if self.pos_embed_mode == 'sinusoidal':
            self.learnable_pos_embedding.requires_grad = False
            if verbose is not None:
                print('learnable positional embedding disabled')

    def enable_classification_parameters(self):
        for param in self.mlp_head.parameters():
            param.requires_grad = True
        if verbose is not None:
            print('mlp_head parameters enabled')
        self.cls_token.requires_grad = True
        if verbose is not None:
            print('cls_token parameters enabled')
        if self.pos_embed_mode == 'learnable':
            self.learnable_pos_embedding.requires_grad = True
            if verbose is not None:
                print('learnable positional embedding enabled')

    def adjust_model(self, num_timesteps, num_channels, sampling_rate, window_duration, num_classes, output, plot=False):
        self.window_duration = window_duration
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length
        self.num_patches = self.num_channels * self.num_windows

        if self.pos_embed_mode == 'learnable':
            learnable_pos_embedding = self.learnable_pos_embedding.unsqueeze(0)
            self.learnable_pos_embedding = F.interpolate(self.learnable_pos_embedding, size=(self.num_patches+1, self.patch_embed_dim), mode='bilinear',
            align_corners=False).squeeze(0)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        patch_embed_projection = self.to_patch_embedding[1].weight.data
        axs[0].imshow(patch_embed_projection.squeeze().squeeze().cpu().detach().numpy())
        axs[0].set_title('pretrained projection matrix before resample')
        d, a, b, t = patch_embed_projection.shape
        resampled_weight = F.interpolate(
            patch_embed_projection, size=(b, self.patch_length), mode='bilinear',
            align_corners=False)
        bias = self.to_patch_embedding[1].bias.data
        self.to_patch_embedding[1] = nn.Conv2d(1, self.patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True)
        self.to_patch_embedding[1].weight.data = resampled_weight
        self.to_patch_embedding[1].bias.data = bias
        axs[1].imshow(self.to_patch_embedding[1].weight.data.squeeze().squeeze().cpu().detach().numpy())
        axs[1].set_title('pretrained projection matrix after resample')
        if plot:
            fig.show()

        # reinitialize classification parameters
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_embed_dim))
        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.patch_embed_dim),
                nn.Linear(self.patch_embed_dim*(2**self.depth), 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.patch_embed_dim * (2 ** self.depth)),
                nn.Linear(self.patch_embed_dim * (2 ** self.depth), num_classes))

    def forward(self, x_dict, *args, **kwargs):
        x = self.to_patch_embedding(x_dict['eeg'])

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed_mode == 'sinusoidal':
            channel_pos = x_dict['channel_voxel_indices']  # batch_size x num_channels
            assert channel_pos.shape[
                       1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
            time_pos = torch.stack([torch.arange(0, self.num_windows, device=x.device, dtype=torch.long) for a in
                                    range(b)])  # batch_size x num_windows  # use sample-relative time positions

            time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)
            channel_pos_embed = self.sinusoidal_pos_embedding(channel_pos).unsqueeze(2).repeat(1, 1, self.num_windows,
                                                                                               1)
            time_pos_embed = rearrange(time_pos_embed, 'b c t d -> b (c t) d')
            channel_pos_embed = rearrange(channel_pos_embed, 'b c t d -> b (c t) d')

            pos_embed = time_pos_embed + channel_pos_embed
            cls_tokens_pos_embedding = repeat(self.learnable_pos_embedding[:, -1, :], '1 d -> b 1 d', b=b)
            pos_embed = torch.concatenate([pos_embed, cls_tokens_pos_embedding], dim=1)

        elif self.pos_embed_mode == 'learnable':
            pos_embed = self.learnable_pos_embedding[:, :(n + 1)]
        else:
            raise ValueError(
                f"pos_embed_mode must be either 'sinusoidal' or 'learnable', but got {self.pos_embed_mode}")

        x += pos_embed
        x = self.dropout(x)

        x, att_matrix = self.encoder(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


    # def forward(self, x, *args, **kwargs):
    #     x = self.to_patch_embedding(x)
    #
    #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    #
    #     b, n, _ = x.shape
    #
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #
    #     if self.pos_embed_mode == 'sinusoidal':
    #         channel_pos = args[-1]  # batch_size x num_channels
    #         assert channel_pos.shape[1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
    #         time_pos = torch.stack([torch.arange(0, self.num_windows, device=x.device, dtype=torch.long) for a in range(b)])  # batch_size x num_windows  # use sample-relative time positions
    #
    #         time_pos_embed = self.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)
    #         channel_pos_embed = self.sinusoidal_pos_embedding(channel_pos).unsqueeze(2).repeat(1, 1, self.num_windows, 1)
    #         time_pos_embed = rearrange(time_pos_embed, 'b c t d -> b (c t) d')
    #         channel_pos_embed = rearrange(channel_pos_embed, 'b c t d -> b (c t) d')
    #
    #         pos_embed = time_pos_embed + channel_pos_embed
    #         cls_tokens_pos_embedding = repeat(self.learnable_pos_embedding[:, -1, :], '1 d -> b 1 d', b=b)
    #         pos_embed = torch.concatenate([pos_embed, cls_tokens_pos_embedding], dim=1)
    #
    #     elif self.pos_embed_mode == 'learnable':
    #         pos_embed = self.learnable_pos_embedding[:, :(n + 1)]
    #     else:
    #         raise ValueError(f"pos_embed_mode must be either 'sinusoidal' or 'learnable', but got {self.pos_embed_mode}")
    #
    #     x += pos_embed
    #     x = self.dropout(x)
    #
    #     x_encoded, encoder_att_matrix = self.encoder(x)
    #     x, decoder_att_matrix = self.decoder(x_encoded)
    #     # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
    #     x = rearrange(x, 'b nt ps -> (b nt) ps')
    #     x = self.to_time_series(x)
    #     x = self.to_latent(x[:, 1:])
    #     return x

    def prepare_data(self, x):
        return x

class HierarchicalAutoTranscoderPretrain(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.5, emb_dropout=0.5, output='multi',
                 pos_embed_mode='learnable', p_t=0.1, p_c=0.2, mask_t_span=2, mask_c_span=5):
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

        self.num_patches = self.num_channels * self.num_windows
        self.hierarchical_autotranscoder = HierarchicalAutoTranscoder(num_timesteps, num_channels, sampling_rate, num_classes,
                                                               depth=depth, num_heads=num_heads,
                                                               feedforward_mlp_dim=feedforward_mlp_dim,
                                                               window_duration=window_duration, pool=pool,
                                                               patch_embed_dim=patch_embed_dim, dim_head=dim_head,
                                                               attn_dropout=attn_dropout, emb_dropout=emb_dropout,
                                                               output=output, pos_embed_mode=pos_embed_mode)
        self.mask_layer = MaskLayer(p_t=p_t, p_c=p_c, c_span=False, mask_t_span=mask_t_span, mask_c_span=mask_c_span,
                                    t_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_channels, self.patch_embed_dim), requires_grad=True),
                                    c_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_windows, self.patch_embed_dim), requires_grad=True))

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (nw ps) -> b (c nw) (ps)', nw=self.num_windows, ps=self.patch_length),
        #     nn.Linear(self.patch_length, patch_embed_dim),
        # )

    def disable_classification_parameters(self):
        self.hierarchical_autotranscoder.disable_classification_parameters()

    def forward(self, x_dict, *args, **kwargs):
        x = self.hierarchical_autotranscoder.to_patch_embedding(x_dict['eeg'])
        x, _, mask_t, mask_c = self.mask_layer(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        b, n, _ = x.shape

        cls_tokens = repeat(self.hierarchical_autotranscoder.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.hierarchical_autotranscoder.pos_embed_mode == 'sinusoidal':
            channel_pos = x_dict['channel_voxel_indices']  # batch_size x num_channels
            assert channel_pos.shape[1] == self.num_channels, "number of channels in meta info and the input tensor's number of channels does not match. when using sinusoidal positional embedding, they must match. This is likely a result of using pca-ica-ed data."
            time_pos = torch.stack([torch.arange(0, self.num_windows, device=x.device, dtype=torch.long) for a in range(b)])  # batch_size x num_windows  # use sample-relative time positions

            time_pos_embed = self.hierarchical_autotranscoder.sinusoidal_pos_embedding(time_pos).unsqueeze(1).repeat(1, self.num_channels, 1, 1)
            channel_pos_embed = self.hierarchical_autotranscoder.sinusoidal_pos_embedding(channel_pos).unsqueeze(2).repeat(1, 1, self.num_windows, 1)
            time_pos_embed = rearrange(time_pos_embed, 'b c t d -> b (c t) d')
            channel_pos_embed = rearrange(channel_pos_embed, 'b c t d -> b (c t) d')

            pos_embed = time_pos_embed + channel_pos_embed
            cls_tokens_pos_embedding = repeat(self.hierarchical_autotranscoder.learnable_pos_embedding[:, -1, :], '1 d -> b 1 d', b=b)
            pos_embed = torch.concatenate([pos_embed, cls_tokens_pos_embedding], dim=1)

        elif self.pos_embed_mode == 'learnable':
            pos_embed = self.hierarchical_autotranscoder.learnable_pos_embedding[:, :(n + 1)]
        else:
            raise ValueError(f"pos_embed_mode must be either 'sinusoidal' or 'learnable', but got {self.pos_embed_mode}")

        x += pos_embed
        x = self.hierarchical_autotranscoder.dropout(x)

        x_encoded, _ = self.hierarchical_autotranscoder.encoder(x)
        x, _ = self.hierarchical_autotranscoder.decoder(x_encoded)
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = rearrange(x[:, 1:], 'b nt ps -> (b nt) ps')
        x = self.hierarchical_autotranscoder.to_time_series(x)
        x = rearrange(x, '(b c w) ps -> b c (w ps)', b=b, c=self.num_channels, w=self.num_windows)
        return x, x_encoded, mask_t, mask_c

    def prepare_data(self, x):
        return x
