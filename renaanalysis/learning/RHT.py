import warnings

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from renaanalysis.learning.HT import MaskLayer
from renaanalysis.models.model_utils import init_weight
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


class PrePostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.post_norm = nn.LayerNorm(dim)
        self.pre_norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.post_norm(self.fn(self.pre_norm(x), **kwargs))


class FeedForwardResidualPostNorm(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.lnorm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.lnorm(self.net(x) + x)


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



class RecurrentGeneralizedPFAttention(nn.Module):
    def __init__(self, embedding_dim, pos_embed_mode, num_heads=8, dim_head=64, drop_attention=0., dropout=0.1, pos_embed_activation=None):
        super().__init__()
        all_heads_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == embedding_dim)

        self.pos_embed_mod = pos_embed_mode
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        self.drop_attention = nn.Dropout(drop_attention)

        self.to_qkv = nn.Linear(embedding_dim, all_heads_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(all_heads_dim, embedding_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.k_r_time_net = nn.Sequential(
            nn.Linear(embedding_dim, all_heads_dim, bias=False),
            # nn.GELU()
        )
        self.k_r_participant_net = nn.Sequential(
            nn.Linear(embedding_dim, all_heads_dim, bias=False),
            # nn.GELU()
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    # def forward(self, x, r_t, r_c, bias_time_e, bias_time_r, bias_channel_r, bias_channel_e):
    def forward(self, x, r_t, r_p, bias_pf, mems, qlen):
        """

        @param x:
        @param r_t:
        @param r_c:
        @param bias_time_e:  num_head x dim_head
        @param bias_time_r:  num_head x dim_head
        @param bias_channel_r:  num_head x dim_head
        @param bias_channel_e:  num_head x dim_head
        @return:
        # """
        b, _,  dpatch = x.shape

        if mems is not None:
            mem_x, mem_r_t, mem_r_p = mems
            if b != mem_x.shape[0]:
                mem_x, mem_r_t, mem_r_p = mem_x[:b], mem_r_t[:b], mem_r_p[:b]
            # x_with_mems = torch.cat([mem_x, x], dim=1)
            klen = x.size(1)
            # x_with_mems = self.layer_norm(torch.cat([mem_x, x], dim=1))
            r_t = torch.cat([mem_r_t, r_t], dim=1) if r_t.size(1) != klen else r_t
            r_p = torch.cat([mem_r_p, r_p], dim=1) if r_p.size(1) != klen else r_t

            qkv = self.to_qkv(x).chunk(3, dim=-1)
            Ex_Wq, Ex_Wke, v = map(lambda t: rearrange(t, 'b n (h d) -> n b h d', h=self.num_heads), qkv)
            Ex_Wq = Ex_Wq[-qlen:]
        else:
            klen = x.size(1)
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # qkv = self.to_qkv(self.layer_norm(x)).chunk(3, dim=-1)
            Ex_Wq, Ex_Wke, v = map(lambda t: rearrange(t, 'b n (h d) -> n b h d', h=self.num_heads), qkv)

        # Ex_Wq = Ex_Wq.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
        # Ex_Wke = Ex_Wke.contiguous().view(ntoken, b, self.num_heads, self.dim_head)
        # v = v.contiguous().view(ntoken, b, self.num_heads, self.dim_head)

        # Ex_Wq = rearrange(Ex_Wq, 'b h n d -> n b h d')
        # Ex_Wke = rearrange(Ex_Wke, 'b h n d -> n b h d')
        # v = rearrange(v, 'b h n d -> n b h d')

        # generalized pf attention ########################################################################
        # repeated_r_p = torch.cat([r_p, repeat(r_p[:, 1:], 'b 1 d-> b k d', k=ntoken - 2)], dim=1)
        # repeated_W_kr_R_p = rearrange(self.k_r_participant_net(repeated_r_p), 'b n (h d)-> n b h d', h=self.num_heads)  # n = 1 + 1 the first is cls token, the second is the participant token, same for all tokens in a sample
        # repeated_BD_p = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, repeated_W_kr_R_p))

        Ex_Wq_e_biased = Ex_Wq + bias_pf  # batch_size, n query, num_heads, dim_head treating bias_time_e as the generalized content bias

        # time relative position bias
        if self.pos_embed_mod == 'sinusoidal':
            W_kr_R_t = rearrange(self.k_r_time_net(r_t), 'b n (h d)-> n b h d', h=self.num_heads)
            W_kr_R_p = rearrange(self.k_r_participant_net(r_p), 'b n (h d)-> n b h d', h=self.num_heads)  # n = 1 + 1 the first is cls token, the second is the participant token, same for all tokens in a sample
        else:
            W_kr_R_t = rearrange(r_t, 'b n h d -> n b h d')
            W_kr_R_p = rearrange(r_p, 'b n h d -> n b h d')

        # without relative shift ######
        # Ex_Wke_r_biased = Ex_Wke + (W_kr_R_t + W_kr_R_c) * 0.5   # batch_size, n query, num_heads, dim_head
        # dots = torch.einsum('ibhd,jbhd->ijbh', Ex_Wq_e_biased, Ex_Wke_r_biased) * self.scale

        # with relative shift ######
        AC = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, Ex_Wke))  # qlen x klen x bsz x n_head
        BD_t = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, W_kr_R_t))
        BD_t = self._rel_shift(BD_t)

        BD_p = torch.einsum('ibnd,jbnd->ijbn', (Ex_Wq_e_biased, W_kr_R_p))
        # repeat the participant token to match the shape of the other tokens
        # BD_p_repeats = repeat(BD_p[:, 1:], 'q 1 b h-> q k b h', k=qlen - 2)  # ntoken - 2, one is the cls token, the other is the participant token being repeated
        # BD_p = torch.cat([BD_p, BD_p_repeats], dim=1)  # ntoken, batch_size, num_heads, dim_head
        BD_p = self._rel_shift(BD_p)

        BD_ = BD_t
        # BD_ = (BD_t + BD_p) # TODO not using participant for now
        # BD_.mul_(1/2)
        dots = AC + BD_ # times 0.5 to normalize across multiple positional features
        dots.mul_(self.scale)  # scale down the dot product

        # transformer-xl attention ########################################################################
        # r = r_t + r_c
        # # W_kr_R_t = self.k_r_time_net(r).view(ntoken, b, self.num_heads, self.dim_head)
        # W_kr_R_t = self.k_r_time_net(r)
        # W_kr_R_t = rearrange(W_kr_R_t, 'b n (h d)-> n b h d', h=self.num_heads)
        #
        # rw_head_q = Ex_Wq + bias_time_e                                         # qlen x bsz x n_head x d_head
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, Ex_Wke))             # qlen x klen x bsz x n_head
        # rr_head_q = Ex_Wq + bias_time_r
        # BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, W_kr_R_t))
        # BD = self._rel_shift(BD)
        #
        # dots = AC + BD
        # dots.mul_(self.scale)
        ##################################################################################################

        attention = self.softmax(dots)
        attention = self.drop_attention(attention)  # n query, n query, batch_size, num_heads

        out = torch.torch.einsum('ijbn,jbnd->ibnd', (attention, v))
        out = rearrange(out, 'n b h d -> b n (h d)')
        
        '''
        # vanilla attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attention = self.softmax(dots)
        attention = self.drop_attention(attention)  # TODO

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        '''

        return self.to_out(out), rearrange(attention, 'q k b h -> b h q k')

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

class RecurrentGeneralizedPFTransformer(nn.Module):
    """
    self.mems
     [  [layerouts, r_t, r_c, r_p] ]
    """
    def __init__(self, embedding_dim, depth, num_heads, dim_head, feedforward_mlp_dim, pos_embed_mode, drop_attention=0., dropout_ff=0.1, mem_len=0):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embedding_dim, RecurrentGeneralizedPFAttention(embedding_dim, pos_embed_mode, num_heads=num_heads, dim_head=dim_head, drop_attention=drop_attention, dropout=dropout_ff)),
                # RecurrentGeneralizedPFAttention(embedding_dim, num_heads=num_heads, dim_head=dim_head, drop_attention=drop_attention, dropout=dropout),
                PreNorm(embedding_dim, FeedForward(embedding_dim, feedforward_mlp_dim, dropout=dropout_ff))  # use pre norm for the attention output residual
            ]))
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.bias_pf = nn.Parameter(torch.randn(self.num_heads, self.dim_head))

        # self.weight_init()
        # list object to store attention results from past forward passes
        self.mems = None
        self.mem_len = mem_len
        self.num_embeds = 3  # x, r_t, r_c, r_p

    def forward(self, x, r_t, r_p):
        """

        @param x:
        @param r_t: relative time positional embedding
        @return:
        """
        if self.mems is None:
            self.init_mems()
        b = x.size(0)
        qlen = x.size(1)
        klen = self.mem_len + qlen
        layer_outs_rs = []
        if self.mem_len > 0: layer_outs_rs.append((x, r_t[:, -qlen:], r_p))
        for i, (attention_layer, prenorm_ff_residual_postnorm) in enumerate(self.layers):
            if self.mems is not None:
                mem_x, _, _, = self.mems[i]
                if b != mem_x.shape[0]:
                    mem_x= mem_x[:b]
                x_with_mems = torch.cat([mem_x, x], dim=1)   # concate x with mem here so they are prenorm together
                out, attention = attention_layer(x_with_mems, r_t=r_t, r_p=r_p, bias_pf=self.bias_pf, mems=self.mems[i], qlen=qlen)
            else:
                out, attention = attention_layer(x, r_t=r_t, r_p=r_p, bias_pf=self.bias_pf, mems=None, qlen=qlen)
            # out, attention = prenorm_attention(x, r_t=r_t, r_c=r_c, bias_time_e=self.bias_time_e, bias_time_r=self.bias_time_r, bias_channel_r=self.bias_channel_r, bias_channel_e=self.bias_channel_e)
            x = out + x  # residual connection
            x = prenorm_ff_residual_postnorm(x) + x

            if self.mem_len > 0: layer_outs_rs.append((x, r_t[:, -qlen:], r_p))  # r_t can be of the same len as t
        self.update_mems(layer_outs_rs, qlen, b)
        return x, attention  # last layer

    def init_mems(self):
        """
        mem will be none if mem_len is 0
        @return:
        """
        if self.mem_len > 0:
            self.mems = []
            param = next(self.parameters())
            for i in range(self.depth + 1):
                empty = [torch.empty(0, dtype=param.dtype, device=param.device) for _ in range (self.num_embeds)]
                self.mems.append(empty)
        else:
            self.mems = None

    def update_mems(self, layer_outs_rs, qlen, b):
        """
        There are `mlen + qlen` steps that can be cached into mems
        For the next step, the last `ext_len` of the `qlen` tokens
        will be used as the extended context. Hence, we only cache
        the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        to `mlen + qlen - self.ext_len`.
        @param layer_outs_rs:
        @param qlen:
        @return:
        """
        if self.mems is None: return
        assert len(layer_outs_rs) == len(self.mems)
        cur_mem_len = self.mems[0][0].size(1) if self.mems[0][0].numel() != 0 else 0
        b_mismatch = len(self.mems[0][0]) - b  # check the batch size
        with torch.no_grad():
            new_mems = []
            end_idx = cur_mem_len + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for layer_index in range(len(layer_outs_rs)):
                # only append mem when 1) mems are empty 2) the number of tokens between the mem and the embeddings matches, when they don't match, it implies the caller is managing the mems for this embedding
                if b_mismatch != 0:
                    layer_outs_rs[layer_index] = [torch.concatenate((self.mems[layer_index][embed_index][-b_mismatch:], layer_outs_rs[layer_index][embed_index]), dim=0)
                                                  for embed_index in range(self.num_embeds)]
                cat = [torch.cat([self.mems[layer_index][embed_index], layer_outs_rs[layer_index][embed_index]], dim=1) for embed_index in range(self.num_embeds)]
                new_mems.append([c[:, beg_idx:end_idx].detach() for c in cat])
        self.mems = new_mems
    # def weight_init(self):
    #     init_weight(self.bias_time_e)
    #     init_weight(self.bias_time_r)
    #     init_weight(self.bias_channel_r)
    #     init_weight(self.bias_channel_e)


class RecurrentPositionalFeatureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class RecurrentHierarchicalTransformer(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, physio_type=eeg_name, depth=4, num_heads=8, feedforward_mlp_dim=32, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.0, emb_dropout=0.1, ff_dropout=0.1, output='multi', n_participant=13, mem_len=1,
                 reset_mem_each_session=False, pos_embed_mode='learnable',
                 window_duration=0.1,
                 token_recep_field = 0.3,
                 token_recep_field_overlap = 0.2,
                 *args, **kwargs):
        """

        # a token is a time slice of data on a single channel

        @param num_timesteps: int: number of timesteps in each sample
        @param num_channels: int: nusmber of channels of the input data
        @param output: str: can be 'single' or 'multi'. If 'single', the output is a single number to be put with sigmoid activation. If 'multi', the output is a vector of size num_classes to be put with softmax activation.
        note that 'single' only works when the number of classes is 2.
        @param reset_mem_each_session:
        """
        if output == 'single':
            assert num_classes == 2, 'output can only be single when num_classes is 2'
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.window_duration = window_duration
        self.physio_type = physio_type

        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.patch_embed_dim = patch_embed_dim
        self.patch_length = int(window_duration * sampling_rate)
        self.num_windows = num_timesteps // self.patch_length
        self.num_patches = self.num_channels * self.num_windows


        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.grid_dims = self.num_channels, self.num_windows
        # self.max_tlen = self.num_windows * (mem_len + 1)  # mem plus the current query
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
        #     nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, self.patch_length), bias=True),
        # )
        t_conv_stride = 10
        self.pool_size = int(sampling_rate * (token_recep_field - window_duration) // t_conv_stride)
        self.pool_stride = (int(sampling_rate * (token_recep_field - window_duration - token_recep_field_overlap)) + self.patch_length)  // t_conv_stride
        self.n_conv_tokens = int((self.num_timesteps - self.patch_length) // t_conv_stride + 1)  # denominator is the stride of the time conv
        self.n_tokens = int((self.n_conv_tokens - self.pool_size) // self.pool_stride + 1)

        self.grid_dims = 1, self.n_tokens

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c t -> b 1 c t', c=self.num_channels, t=self.num_timesteps),
            nn.Conv2d(1, patch_embed_dim, kernel_size=(1, self.patch_length), stride=(1, t_conv_stride), bias=True),
            nn.Conv2d(patch_embed_dim, patch_embed_dim, (self.num_channels, 1), (1, 1)),
            nn.BatchNorm2d(patch_embed_dim),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size), (1, self.pool_stride)),
            nn.Dropout(emb_dropout),
        )

        self.pos_embed_mode = pos_embed_mode
        if self.pos_embed_mode == 'sinusoidal':
            self.pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)
        elif self.pos_embed_mode == 'learnable':
            self.learnable_time_embedding = nn.Parameter(torch.randn(1, self.n_tokens, num_heads, dim_head))
            # self.learnable_channel_embedding = nn.Parameter(torch.randn(1, self.num_channels, num_heads, dim_head))
            self.learnable_participant_embedding_list = nn.Parameter(torch.randn(n_participant, 1, 1, num_heads, dim_head)) # keeps embedding for every participant

        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.cls_token_pos_embedding = nn.Parameter(torch.randn(1, 1, patch_embed_dim )) if pos_embed_mode == 'sinusoidal' else nn.Parameter(torch.randn(1, 1, num_heads, dim_head))
        self.dropout = nn.Dropout(emb_dropout)

        self.mem_len = (self.n_tokens * mem_len + 1 ) if mem_len != 0 else 0  # +1 for cls token
        self.transformer = RecurrentGeneralizedPFTransformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, pos_embed_mode, drop_attention=attn_dropout, dropout_ff=ff_dropout, mem_len=self.mem_len)

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

        randomized_participant_pos = torch.randperm(n_participant).to(torch.float32)
        self.register_buffer('randomized_participant_pos', randomized_participant_pos)

        self.current_session = None
        self.reset_mem_each_session = reset_mem_each_session

    def forward(self, x, *args, **kwargs):
        """
        auditory oddball meta info: 'subject_id', 'run', 'epoch_start_times', 'channel_positions', 'channel_voxel_indices'
        @param x_eeg:
        @param meta_info: meta_info is a dictionary
        @return:
        """

        # check meta info is complete
        x = self.encode(x, *args, **kwargs)
        return self.mlp_head(x)

    def encode(self, x, *args, **kwargs):
        """
        currently

        @param x_eeg:
        @param args:
        @param kwargs:
        @return:
        """
        x_eeg = x[self.physio_type] if type(x) is dict else x

        b, nchannel, _ = x_eeg.shape

        if self.reset_mem_each_session and self.current_session != (current_session := x['session'][0].item()):
            self.reset()
            print(f"Current session changed from {self.current_session} to {current_session}. Memory reset.")
            self.current_session = current_session

        # get discretized time for each token
        # discretized_start_times = args[3]  // self.window_duration
        mem_timesteps = int(self.transformer.mems[0][0].size(1) / self.num_channels) if (self.transformer.mems is not None and torch.numel(self.transformer.mems[0][0]) != 0) else 0
        mem_num_epochs = mem_timesteps // self.num_windows
        tlen = mem_timesteps + self.n_tokens

        participant_pos = torch.unique(x['subject_id']).to(int)
        assert len(torch.unique(x['subject_id']) == 0), 'only one subject per batch is supported'

        if self.pos_embed_mode == 'sinusoidal':
            # time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
            # time_pos = torch.stack([torch.arange(a, a+self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use session-relative time positions
            # time_pos = torch.stack([torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
            time_pos = torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long)[None, :]  # batch_size x num_windows  # use sample-relative time positions

            # compute channel positions that are voxel discretized
            channel_pos = x['channel_voxel_indices']  # batch_size x num_channels

            # get the participant embeddings
            participant_pos = self.randomized_participant_pos[participant_pos][:, None]

            # each sample in a batch must have the same participant embedding
            time_pos_embed = self.pos_embedding(time_pos)
            # channel_pos_embed = self.pos_embedding(channel_pos)
            participant_pos_embed = self.pos_embedding(participant_pos)  # no need to repeat because every token in the same sample has the same participant embedding
            # time_pos_embed = time_pos_embed.unsqueeze(1).repeat(b, 1, 1).reshape(b, -1, self.patch_embed_dim)
            # channel_pos_embed = channel_pos_embed.unsqueeze(2).repeat(1, 1, self.num_windows, 1).reshape(b, -1,self.patch_embed_dim)
            participant_pos_embed = repeat(participant_pos_embed, '1 1 d-> b n d', b=b, n=self.n_tokens)
            time_pos_embed = repeat(time_pos_embed, '1 n d-> b n d', b=b)  # repeat for batches
        else:  # learnable
            raise NotImplementedError
            # time_pos_embed = self.learnable_time_embedding[:, -tlen:]
            # channel_pos_embed = self.learnable_channel_embedding
            # participant_pos_embed = repeat(self.learnable_participant_embedding_list[int(participant_pos)], '1 1 h d -> b n h d', b=b, n=self.num_patches)  # repeat for batch
            #
            # time_pos_embed = repeat(time_pos_embed.unsqueeze(1), '1 1 t h d -> b (c t) h d', b=b, c=nchannel)
            # channel_pos_embed = repeat(channel_pos_embed.unsqueeze(2), '1 c 1 h d -> b (c t) h d', b=b, t=self.num_windows)

        # viz_time_positional_embedding(time_pos_embed)  # time embedding differs in batch
        # viz_time_positional_embedding(channel_pos_embed)  # time embedding differs in batch

        # prepare the positional features that are different among tokens in the same sample

        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BDHW -> BND  D is the embedding dimension, or channels in conv language

        b, ntoken, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed_mode == 'sinusoidal':
            cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 d -> b 1 d', b=b)
        else:
            cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 h d -> b 1 h d', b=b)

        # insert cls pos embedding based on the number of mem steps
        for i in range(mem_num_epochs + 1):
            time_pos_embed = torch.cat((time_pos_embed[:, :i * ntoken, :], cls_tokens_pos_embedding, time_pos_embed[:, i * ntoken:, :]), dim=1)  # only time pos embedding can be of klen before the transformer
        # channel_pos_embed = torch.cat((cls_tokens_pos_embedding, channel_pos_embed), dim=1)
        participant_pos_embed = torch.cat((cls_tokens_pos_embedding, participant_pos_embed), dim=1)

        # time_pos_embed = self.dropout(time_pos_embed)
        # channel_pos_embed = self.dropout(channel_pos_embed)

        # x += self.learnablepos_embedding[:, :(ntoken + 1)]

        # x += time_pos_embed
        # x += channel_pos_embed

        x, att_matrix = self.transformer(x, time_pos_embed, participant_pos_embed)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x


    def reset(self):
        # warnings.warn("RHT.reset(): To be implemented")
        self.transformer.init_mems()

class RecurrentHierarchicalTransformerAutoEncoderPretrain(nn.Module):
    def __init__(self, num_timesteps, num_channels, sampling_rate, num_classes, depth=4, num_heads=8, feedforward_mlp_dim=32, window_duration=0.1, pool='cls',
                 patch_embed_dim=128, dim_head=64, attn_dropout=0.0, emb_dropout=0.1, dropout=0.1, pos_embed_mode='learnable', output='multi', n_participant=5000, mem_len=1, p_t=0.7, p_c=0.7):
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

        self.learnablepos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))
        self.pos_embedding = SinusoidalPositionalEmbedding(patch_embed_dim)
        self.learnable_p_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.cls_token_pos_embedding = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.mem_len = (self.num_channels * self.num_windows * mem_len + 1 ) if mem_len != 0 else 0  # +1 for cls token
        self.transformer = RecurrentGeneralizedPFTransformer(patch_embed_dim, depth, num_heads, dim_head, feedforward_mlp_dim, drop_attention=attn_dropout, dropout_ff=dropout, mem_len=self.mem_len, pos_embed_mode=pos_embed_mode)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mask_layer = MaskLayer(p_t=p_t, p_c=p_c, c_span=False, mask_t_span=1, mask_c_span=1,
                                    t_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_channels, self.patch_embed_dim), requires_grad=True),
                                    c_mask_replacement=torch.nn.Parameter(
                                        torch.zeros(self.num_windows, self.patch_embed_dim), requires_grad=True),
                                    is_constant_size=True)

        if output == 'single':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, 1))
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(patch_embed_dim),
                nn.Linear(patch_embed_dim, num_classes))

        randomized_participant_pos = torch.randperm(n_participant).to(torch.float32)
        self.register_buffer('randomized_participant_pos', randomized_participant_pos)

    def disable_classification_parameters(self):
        # self.transformer.disable_classification_parameters
        pass

    def forward(self, x_eeg, *args, **kwargs):
        """
        auditory oddball meta info: 'subject_id', 'run', 'epoch_start_times', 'channel_positions', 'channel_voxel_indices'
        @param x_eeg:
        @param meta_info: meta_info is a dictionary
        @return:
        """

        # check meta info is complete
        x = self.encode(x_eeg, *args, **kwargs)
        return self.mlp_head(x)

    def encode(self, x, *args, **kwargs):
        x_eeg = x['eeg']

        b, nchannel, _ = x_eeg.shape

        if self.reset_mem_each_session and self.current_session != (current_session := x['session'][0].item()):
            self.reset()
            print(f"Current session changed from {self.current_session} to {current_session}. Memory reset.")
            self.current_session = current_session

        # get discretized time for each token
        # discretized_start_times = args[3]  // self.window_duration
        mem_timesteps = int(self.transformer.mems[0][0].size(1) / self.num_channels) if (
                    self.transformer.mems is not None and torch.numel(self.transformer.mems[0][0]) != 0) else 0
        mem_num_epochs = mem_timesteps // self.num_windows
        tlen = mem_timesteps + self.num_windows

        participant_pos = torch.unique(x['subject_id']).item()

        if self.pos_embed_mode == 'sinusoidal':
            # time_pos = torch.stack([torch.arange(0, self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
            # time_pos = torch.stack([torch.arange(a, a+self.num_windows, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use session-relative time positions
            # time_pos = torch.stack([torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long) for a in discretized_start_times])  # batch_size x num_windows  # use sample-relative time positions
            time_pos = torch.arange(0, tlen, device=x_eeg.device, dtype=torch.long)[None,
                       :]  # batch_size x num_windows  # use sample-relative time positions

            # compute channel positions that are voxel discretized
            channel_pos = x['channel_voxel_indices']  # batch_size x num_channels

            # get the participant embeddings
            participant_pos = self.randomized_participant_pos[participant_pos][:,
                              None]  # batch_size x 1, get random participant pos

            # each sample in a batch must have the same participant embedding
            time_pos_embed = self.pos_embedding(time_pos)
            channel_pos_embed = self.pos_embedding(channel_pos)
            participant_pos_embed = self.pos_embedding(
                participant_pos)  # no need to repeat because every token in the same sample has the same participant embedding

            time_pos_embed = time_pos_embed.unsqueeze(1).repeat(b, nchannel, 1, 1).reshape(b, -1, self.patch_embed_dim)
            channel_pos_embed = channel_pos_embed.unsqueeze(2).repeat(1, 1, self.num_windows, 1).reshape(b, -1,
                                                                                                         self.patch_embed_dim)
            participant_pos_embed = repeat(participant_pos_embed, 'b 1 d-> b n d', n=self.num_patches)
        else:  # learnable
            time_pos_embed = self.learnable_time_embedding[:, -tlen:]
            channel_pos_embed = self.learnable_channel_embedding
            participant_pos_embed = repeat(self.learnable_participant_embedding_list[int(participant_pos)],
                                           '1 1 h d -> b n h d', b=b, n=self.num_patches)  # repeat for batch

            time_pos_embed = repeat(time_pos_embed.unsqueeze(1), '1 1 t h d -> b (c t) h d', b=b, c=nchannel)
            channel_pos_embed = repeat(channel_pos_embed.unsqueeze(2), '1 c 1 h d -> b (c t) h d', b=b,
                                       t=self.num_windows)

        b, nchannel, _ = x_eeg.shape


        x = self.to_patch_embedding(x_eeg)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        b, ntoken, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        cls_tokens_pos_embedding = repeat(self.cls_token_pos_embedding, '1 1 d -> b 1 d', b=b)

        # insert cls pos embedding based on the number of mem steps
        for i in range(mem_num_epochs + 1):
            time_pos_embed = torch.cat((time_pos_embed[:, :i * ntoken, :], cls_tokens_pos_embedding, time_pos_embed[:, i * ntoken:, :]), dim=1)  # only time pos embedding can be of klen before the transformer

        channel_pos_embed = torch.cat((cls_tokens_pos_embedding, channel_pos_embed), dim=1)
        participant_pos_embed = torch.cat((cls_tokens_pos_embedding, participant_pos_embed), dim=1)

        # time_pos_embed = self.dropout(time_pos_embed)
        # channel_pos_embed = self.dropout(channel_pos_embed)

        # x += self.learnablepos_embedding[:, :(ntoken + 1)]

        # x += time_pos_embed
        # x += channel_pos_embed

        x = self.dropout(x)

        x, att_matrix = self.transformer(x, time_pos_embed, channel_pos_embed, participant_pos_embed)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x


    def reset(self):
        # warnings.warn("RHT.reset(): To be implemented")
        self.transformer.init_mems()