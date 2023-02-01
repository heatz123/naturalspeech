import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from utils import commons
from utils.commons import init_weights, get_padding
from models import modules
from models import attentions
from models.modules import ConvBlock, SwishBlock, LinearNorm


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask)

        return x, x_mask


class LearnableUpsampling(nn.Module):
    def __init__(
        self,
        d_predictor=192,
        kernel_size=3,
        dropout=0.0,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super(LearnableUpsampling, self).__init__()
        self.max_seq_len = max_seq_len

        # Attention (W)
        self.conv_w = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_w = SwishBlock(conv_output_size + 2, dim_w, dim_w)
        self.linear_w = LinearNorm(dim_w * d_predictor, d_predictor, bias=True)
        self.softmax_w = nn.Softmax(dim=2)

        # Auxiliary Attention Context (C)
        self.conv_c = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_c = SwishBlock(conv_output_size + 2, dim_c, dim_c)

        # Upsampled Representation (O)
        self.linear_einsum = LinearNorm(dim_c * dim_w, d_predictor)  # A
        self.layer_norm = nn.LayerNorm(d_predictor)

        self.proj_o = LinearNorm(192, 192 * 2)

    def forward(self, duration, V, src_len, src_mask, max_src_len):

        batch_size = duration.shape[0]

        # Duration Interpretation
        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(V.device)
        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len)

        # Prepare Attention Mask
        src_mask_ = src_mask.unsqueeze(1).expand(
            -1, mel_mask.shape[1], -1
        )  # [B, tat_len, src_len]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(
            -1, -1, src_mask.shape[1]
        )  # [B, tgt_len, src_len]
        attn_mask = torch.zeros(
            (src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])
        ).to(V.device)
        attn_mask = attn_mask.masked_fill(src_mask_, 1.0)
        attn_mask = attn_mask.masked_fill(mel_mask_, 1.0)
        attn_mask = attn_mask.bool()

        # Token Boundary Grid
        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration
        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        t_arange = (
            torch.arange(1, max_mel_len + 1, device=V.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, max_src_len)
        )

        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(
            attn_mask, 0
        )

        # Attention (W)
        W = self.swish_w(S, E, self.conv_w(V))  # [B, T, K, dim_w]
        W = W.masked_fill(src_mask_.unsqueeze(-1), -np.inf)
        W = self.softmax_w(W)  # [B, T, K]
        W = W.masked_fill(mel_mask_.unsqueeze(-1), 0.0)
        W = W.permute(0, 3, 1, 2)

        # Auxiliary Attention Context (C)
        C = self.swish_c(S, E, self.conv_c(V))  # [B, T, K, dim_c]

        # Upsampled Representation (O)
        upsampled_rep = self.linear_w(
            torch.einsum("bqtk,bkh->bqth", W, V).permute(0, 2, 1, 3).flatten(2)
        ) + self.linear_einsum(
            torch.einsum("bqtk,btkp->bqtp", W, C).permute(0, 2, 1, 3).flatten(2)
        )  # [B, T, M]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)
        upsampled_rep = self.proj_o(upsampled_rep)

        return upsampled_rep, mel_mask, mel_len, W

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class VAEMemoryBank(nn.Module):
    def __init__(
        self, bank_size=1000, n_hidden_dims=192, n_attn_heads=2, init_values=None
    ):
        super().__init__()

        self.bank_size = bank_size
        self.n_hidden_dims = n_hidden_dims
        self.n_attn_heads = n_attn_heads

        self.encoder = attentions.MultiHeadAttention(
            channels=n_hidden_dims,
            out_channels=n_hidden_dims,
            n_heads=n_attn_heads,
        )

        self.memory_bank = nn.Parameter(torch.randn(n_hidden_dims, bank_size))
        if init_values is not None:
            with torch.no_grad():
                self.memory_bank.copy_(init_values)

    def forward(self, z):
        b, _, _ = z.shape
        return self.encoder(
            z, self.memory_bank.unsqueeze(0).repeat(b, 1, 1), attn_mask=None
        )


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat=None):
        if y_hat is None:
            return self.forward_y_hat(y)
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def forward_y_hat(self, y_hat):
        y_d_gs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_g, fmap_g = d(y_hat)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_gs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, n_vocab, spec_channels, segment_size, hps_models, **kwargs):

        super().__init__()
        self.segment_size = segment_size

        self.enc_p = TextEncoder(n_vocab=n_vocab, **hps_models.phoneme_encoder)
        self.dec = Generator(**hps_models.decoder)

        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels, **hps_models.posterior_encoder
        )
        self.flow = ResidualCouplingBlock(**hps_models.flow)

        self.dp = DurationPredictor(**hps_models.duration_predictor)
        self.learnable_upsampling = LearnableUpsampling(
            **hps_models.learnable_upsampling
        )

        self.use_memory_bank = False

    def forward(self, x, x_lengths, y, y_lengths, d=None, use_gt_duration=True):

        # text encoder
        x, x_mask = self.enc_p(x, x_lengths)

        # prior encoder & flow
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=None)
        z_p = self.flow(z, y_mask, g=None)

        # differentiable durator (duration predictor & loss)
        logw = self.dp(x, x_mask, g=None)
        w = torch.exp(logw) * x_mask

        w_ = d.unsqueeze(1)
        logw_ = torch.log(w_ + 1e-6) * x_mask

        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        if not use_gt_duration:
            d = w.squeeze(1)  # use predicted duration

        # differentiable durator (learnable upsampling)
        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            d,
            x.transpose(1, 2),
            x_lengths,
            ~(x_mask.squeeze(1).bool()),
            x_lengths.max(),
        )
        p_mask = ~p_mask

        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        if self.use_memory_bank:
            z_slice = self.memory_bank(z_slice)

        o = self.dec(z_slice, g=None)

        z_q = self.flow(
            m_p + torch.randn_like(m_p) * torch.exp(logs_p),
            p_mask.unsqueeze(1),
            g=None,
            reverse=True,
        )
        z_q_lengths = p_mask.flatten(1, -1).sum(dim=-1).long()
        z_slice_q, ids_slice_q = commons.rand_slice_segments(
            z_q, torch.minimum(z_q_lengths, y_lengths), self.segment_size
        )

        if self.use_memory_bank:
            z_slice_q = self.memory_bank(z_slice_q)

        o2 = self.dec((z_slice_q), g=None)

        return (
            o,
            l_length,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q, p_mask),
            W,
            o2,
            z_q,
            d,
            ids_slice_q,
        )

    def infer(self, x, x_lengths, noise_scale=1, length_scale=1, max_len=None, d=None):
        # infer with only one example
        x, x_mask = self.enc_p(x, x_lengths)

        logw = self.dp(x, x_mask, g=None)
        w = torch.exp(logw) * x_mask * length_scale
        if d is not None:
            w = d.unsqueeze(1) * x_mask * length_scale

        upsampled_rep, p_mask, _, W = self.learnable_upsampling(
            w.squeeze(1),
            x.transpose(1, 2),
            x_lengths,
            ~(x_mask.squeeze(1).bool()),
            x_mask.shape[-1],
        )
        p_mask = ~p_mask
        m_p, logs_p = torch.split(upsampled_rep.transpose(1, 2), 192, dim=1)

        y_mask = p_mask.unsqueeze(1)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=None, reverse=True)

        if self.use_memory_bank:
            z = self.memory_bank(z)

        o = self.dec((z * y_mask)[:, :, :max_len], g=None)
        return o, y_mask, (z, z_p, m_p, logs_p)

    def attach_memory_bank(self, hps_models):
        device = next(self.parameters()).device

        self.memory_bank = VAEMemoryBank(**hps_models.memory_bank).to(device)
        self.use_memory_bank = True
