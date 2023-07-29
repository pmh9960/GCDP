import math
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from random import random
from typing import List, Union

import kornia.augmentation as K
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import check_shape, rearrange_many
from einops_exts.torch import EinopsToAndFrom
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from torch.special import expm1

from imagen_pytorch.imagen_pytorch import (
    Attention, CrossEmbedLayer, Downsample, GaussianDiffusionContinuousTimes,
    Identity, LearnedSinusoidalPosEmb, LinearAttentionTransformerBlock,
    NullUnet, Parallel, PerceiverResampler, PixelShuffleUpsample,
    Residual, ResnetBlock, TransformerBlock, Upsample, UpsampleCombiner,
    cast_tuple, cast_uint8_images_to_float, default, eval_decorator,
    exists, first, identity, is_float_dtype, maybe, module_device,
    normalize_neg_one_to_one, pad_tuple_to_length, print_once, prob_mask_like,
    resize_image_to, right_pad_dims_to, unnormalize_zero_to_one, zero_init_,
    beta_linear_log_snr, alpha_cosine_log_snr, log, log_snr_to_alpha_sigma)
from imagen_pytorch.imagen_video.imagen_video import Unet3D, resize_video_to
from imagen_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim, t5_encode_text


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    if len(x.size()) == 4 and x.size(1) == 1:
        x = x.squeeze(1)
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1, keepdims=True)


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


@torch.jit.script
def alpha_cosine_p_log_snr(t, p: float = 0.8, s: float = 0.008):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** (-2 * p)) - 1, eps=1e-5)


class MultinomialDiffusion(nn.Module):
    def __init__(self, num_classes, *, noise_schedule, p=1.0, timesteps=1000):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == "cosine_p":
            self.log_snr = partial(alpha_cosine_p_log_snr, p=p)
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        cumprod_alpha = torch.tensor([log_snr_to_alpha_sigma(self.log_snr(
            torch.tensor(t / timesteps)))[0] ** 2 for t in range(timesteps)])
        alphas = cumprod_alpha / F.pad(cumprod_alpha, (1, 0), value=cumprod_alpha[0])[:-1]
        self.register_buffer('log_alpha', torch.log(alphas))
        self.register_buffer('log_1_min_alpha', log_1_min_a(self.log_alpha))
        self.register_buffer('log_cumprod_alpha', torch.cumsum(self.log_alpha, axis=0))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_a(self.log_cumprod_alpha))
        self.num_classes = num_classes
        self.num_timesteps = timesteps

    def change_times_dtype(self, t):
        if t.dtype != torch.int64:
            if ((t < 1) * (t > 0)).any():
                return torch.floor(t * self.num_timesteps).to(torch.int64)
            else:
                return t.to(torch.int64)
        return t

    def get_times(self, batch_size, noise_level, *, device):
        raise NotImplementedError
        return torch.full((batch_size,), noise_level, device=device, dtype=torch.float32)

    def sample_random_times(self, batch_size, max_thres=0.999, *, device):
        raise NotImplementedError
        return torch.zeros((batch_size,), device=device).float().uniform_(0, max_thres)

    def get_condition(self, times):
        raise NotImplementedError
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        raise NotImplementedError
        times = torch.linspace(1., 0., self.num_timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - math.log(self.num_classes)
        )
        return log_probs

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        t = self.change_times_dtype(t)  # caused by continuous timesteps.
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - math.log(self.num_classes)
        )
        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        t = self.change_times_dtype(t)  # caused by continuous timesteps.
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def q_sample_from_to(self, log_x_from, from_t, to_t):
        shape, device, dtype = log_x_from.shape, log_x_from.device, log_x_from.dtype
        batch = shape[0]

        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device=device, dtype=dtype)

        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device=device, dtype=dtype)

        from_t = self.change_times_dtype(from_t)  # caused by continuous timesteps.
        to_t = self.change_times_dtype(to_t)  # caused by continuous timesteps.

        log_cumprod_alpha_to_t = extract(self.log_cumprod_alpha, to_t, log_x_from.shape)
        log_cumprod_alpha_from_t = extract(self.log_cumprod_alpha, from_t, log_x_from.shape)
        log_probs = log_add_exp(
            log_x_from + log_cumprod_alpha_to_t - log_cumprod_alpha_from_t,
            log_1_min_a(log_cumprod_alpha_to_t - log_cumprod_alpha_from_t) - math.log(self.num_classes)
        )

        mask = (to_t == torch.zeros_like(to_t)).float()[:, None, None, None]
        log_sample = index_to_log_onehot(log_probs.argmax(dim=1), self.num_classes) * mask \
            + self.log_sample_categorical(log_probs) * (1. - mask)

        return log_sample

    def predict_start_from_noise(self, x_t, t, noise):
        raise NotImplementedError

    # calculate loss

    def multinomial_kl(self, log_prob1, log_prob2):
        return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def loss_fn(self, target_log_lbl, pred_lbl, t, log_lbl):
        t = self.change_times_dtype(t)
        pt = torch.ones_like(t).float() / self.num_timesteps

        kl = self.multinomial_kl(target_log_lbl, pred_lbl)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_lbl, pred_lbl)
        decoder_nll = sum_except_batch(decoder_nll)
        mask = (t == torch.zeros_like(t)).float()
        kl = mask * decoder_nll + (1. - mask) * kl

        kl_prior = self.kl_prior(log_lbl)
        vb_loss = kl / pt + kl_prior

        loss = vb_loss / (math.log(2) * pred_lbl.shape[1:].numel())
        return loss


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        self.emb_layer = nn.Embedding(num_classes, channels)

    def forward(self, x):
        assert x.dim() == 4, f'x.shape should be (B, 1, H, W) but {x.shape}'
        assert x.size(1) == 1, f'x.shape should be (B, 1, H, W) but {x.shape}'
        x = self.emb_layer(x.long().squeeze(1))
        x = x.permute(0, 3, 1, 2)
        return x


class JointUnet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_classes,
        image_embed_dim=1024,
        text_embed_dim=get_encoded_dim(DEFAULT_T5_NAME),
        num_resnet_blocks=1,
        cond_dim=None,
        num_image_tokens=4,
        num_time_tokens=2,
        learned_sinu_pos_emb_dim=16,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        cond_images_channels=0,
        channels=3,
        channels_lbl=3,
        channels_out=None,
        attn_dim_head=64,
        attn_heads=8,
        ff_mult=2.,
        lowres_cond=False,                # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns=True,
        layer_attns_depth=1,
        # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        layer_attns_add_text_cond=True,
        # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        attend_at_middle=True,
        layer_cross_attns=True,
        use_linear_attn=False,
        use_linear_cross_attn=False,
        cond_on_text=True,
        max_text_len=256,
        init_dim=None,
        resnet_groups=8,
        init_conv_kernel_size=7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed=True,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        attn_pool_text=True,
        attn_pool_num_latents=32,
        dropout=0.,
        memory_efficient=False,
        init_conv_to_final_conv_residual=False,
        use_global_context_attn=True,
        scale_skip_connection=True,
        final_resnet_block=True,
        final_conv_kernel_size=3,
        cosine_sim_attn=False,
        self_cond=False,
        combine_upsample_fmaps=False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample=True        # may address checkboard artifacts
    ):
        super().__init__()

        # guide researchers

        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # label embedding

        self.num_classes = num_classes
        self.init_emb_seg = LabelEmbedding(self.num_classes, channels_lbl)
        self.init_emb_seg_lowres = LabelEmbedding(self.num_classes, channels_lbl) if lowres_cond else None

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)
        # (3) in joint diffusion, label condition appends on image.
        init_channels = (channels + channels_lbl) * (1 + int(lowres_cond) + int(self_cond))  # Joint Imagen
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond
        if self_cond:
            self.self_cond_lbl_emb = LabelEmbedding(self.num_classes, channels_lbl)

        # optional image conditioning

        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        init_channels += cond_images_channels

        # initial convolution

        self.init_conv = CrossEmbedLayer(init_channels, dim_out=init_dim, kernel_sizes=init_cross_embed_kernel_sizes, stride=1) if init_cross_embed else nn.Conv2d(
            init_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # embedding time for log(snr) noise from continuous version

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1

        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r=num_time_tokens)
        )

        # low res aug noise conditioning

        self.lowres_cond = lowres_cond

        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r=num_time_tokens)
            )

        # normalizations

        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = PerceiverResampler(dim=cond_dim, depth=2, dim_head=attn_dim_head, heads=attn_heads,
                                            num_latents=attn_pool_num_latents, cosine_sim_attn=cosine_sim_attn) if attn_pool_text else None

        # for classifier free guidance

        self.max_text_len = max_text_len

        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # for non-attention based text conditioning at all points in the network where time is also conditioned

        self.to_text_non_attn_cond = None

        if cond_on_text:
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

        # attention related params

        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head, cosine_sim_attn=cosine_sim_attn)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass

        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim=time_cond_dim,
                                              groups=resnet_groups[0], use_gca=use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, resnet_groups, layer_attns,
                        layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = []  # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(
                    nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, cond_dim=layer_cond_dim,
                             linear_attn=layer_use_linear_cross_attn, time_cond_dim=time_cond_dim, groups=groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim=time_cond_dim,
                              groups=groups, use_gca=use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim=current_dim, depth=layer_attn_depth,
                                        ff_mult=ff_mult, context_dim=cond_dim, **attn_kwargs),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim,
                                      time_cond_dim=time_cond_dim, groups=resnet_groups[-1])
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(
            Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim,
                                      time_cond_dim=time_cond_dim, groups=resnet_groups[-1])

        # upsample klass

        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers

        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim=layer_cond_dim,
                             linear_attn=layer_use_linear_cross_attn, time_cond_dim=time_cond_dim, groups=groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim=time_cond_dim,
                              groups=groups, use_gca=use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim=dim_out, depth=layer_attn_depth, ff_mult=ff_mult,
                                        context_dim=cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity(),
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out

        self.upsample_combiner = UpsampleCombiner(
            dim=dim,
            enabled=combine_upsample_fmaps,
            dim_ins=upsample_fmap_dims,
            dim_outs=dim
        )

        # whether to do a final residual from initial conv to the final resnet block out

        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out

        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim=time_cond_dim,
                                           groups=resnet_groups[0], use_gca=True) if final_resnet_block else None

        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        final_conv_dim_in += (channels + channels_lbl) if lowres_cond else 0

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out,
                                    final_conv_kernel_size, padding=final_conv_kernel_size // 2)
        self.final_conv_seg = nn.Conv2d(final_conv_dim_in, self.num_classes,
                                        final_conv_kernel_size, padding=final_conv_kernel_size // 2)

        zero_init_(self.final_conv)
        zero_init_(self.final_conv_seg)

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        if lowres_cond == self.lowres_cond and \
                channels == self.channels and \
                cond_on_text == self.cond_on_text and \
                text_embed_dim == self._locals['text_embed_dim'] and \
                channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
            cond_on_text=cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # methods for returning the full unet config as well as its parameter state

    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # class method for rehydrating the unet from its config and state dict

    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # methods for persisting unet to disk

    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok=True, parents=True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config=config, state_dict=state_dict)
        torch.save(pkg, str(path))

    # class method for rehydrating the unet from file saved with `persist_to_file`

    @classmethod
    def hydrate_from_file(klass, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        assert 'config' in pkg and 'state_dict' in pkg
        config, state_dict = pkg['config'], pkg['state_dict']

        return JointUnet.from_config_and_state_dict(config, state_dict)

    # forward with classifier free guidance

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.,
        **kwargs
    ):
        logits, logits_seg = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits, logits_seg

        null_logits, null_logits_seg = self.forward(*args, cond_drop_prob=1., **kwargs)

        cond_logits = null_logits + (logits - null_logits) * cond_scale

        # TODO: CFG of categorical is not clear.
        cond_logits_seg = null_logits_seg + (logits_seg - null_logits_seg) * cond_scale

        return cond_logits, cond_logits_seg

    def forward(
        self,
        x,
        lbl,
        time,
        *,
        lowres_cond_img=None,
        lowres_cond_lbl=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        self_cond=None,
        self_cond_lbl=None,
        cond_drop_prob=0.
    ):
        batch_size, device = x.shape[0], x.device

        # joint imagen

        lbl = self.init_emb_seg(lbl.long())
        x = torch.cat((x, lbl), dim=1)

        # condition on self

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            if self_cond_lbl is None:
                self_cond_lbl = torch.zeros_like(lbl)
            else:
                self_cond_lbl = self.self_cond_lbl_emb(self_cond_lbl.long())
            x = torch.cat((x, self_cond, self_cond_lbl), dim=1)

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)
                    ), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)
                    ), 'low resolution conditioning noise time must be present'

        if exists(lowres_cond_img) and exists(lowres_cond_lbl):
            lowres_cond_lbl = self.init_emb_seg_lowres(lowres_cond_lbl.long())
            x = torch.cat((x, lowres_cond_img, lowres_cond_lbl), dim=1)

        # condition on input image

        assert not (self.has_cond_image ^ exists(cond_images)), \
            'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        if exists(cond_images):
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            cond_images = resize_image_to(cond_images, x.shape[-1])
            x = torch.cat((cond_images, x), dim=1)

        # initial convolution

        x = self.init_conv(x)

        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim=-2)

        # text conditioning

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:

            # conditional dropout

            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)

            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            # calculate text embeds

            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, :self.max_text_len]

            if exists(text_mask):
                text_mask = text_mask[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            null_text_embed = self.null_text_embed.to(text_tokens.dtype)  # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens

            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            null_text_hidden = self.null_text_hidden.to(t.dtype)

            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            t = t + text_hiddens

        # main conditioning tokens (c)

        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim=-2)

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        def add_skip_connection(x): return torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())

            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        if exists(lowres_cond_img) and exists(lowres_cond_lbl):
            x = torch.cat((x, lowres_cond_img, lowres_cond_lbl), dim=1)

        return self.final_conv(x), self.final_conv_seg(x)

# predefined unets, with configs lining up with hyperparameters in appendix of paper


class BaseJointUnet(JointUnet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, False, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})


class SRJointUnet(JointUnet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

# main imagen ddpm class, which is a cascading DDPM from Ho et al.


class JointImagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        num_classes,
        text_encoder_name=DEFAULT_T5_NAME,
        text_embed_dim=None,
        channels=3,
        timesteps=1000,
        sample_timesteps=100,
        cond_drop_prob=0.1,
        loss_type='l2',
        noise_schedules='cosine',
        noise_schedules_lbl='cosine_p',
        cosine_p_lbl=1.0,
        pred_objectives='noise',
        random_crop_sizes=None,
        lowres_noise_schedule='linear',
        # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        lowres_sample_noise_level=0.2,
        # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        per_sample_random_aug_noise_level=False,
        lowres_max_thres=0.999,
        condition_on_text=True,
        # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        auto_normalize_img=True,
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_gamma=0.5,
        p2_loss_weight_k=1,
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.95,     # unsure what this was based on perusal of paper
        only_train_unet_number=None,
    ):
        super().__init__()

        # joint

        self.num_classes = num_classes

        # loss

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # conditioning hparams

        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # determine noise schedules per unet

        timesteps = cast_tuple(timesteps, num_unets)
        sample_timesteps = cast_tuple(sample_timesteps, num_unets)

        # make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets

        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')
        noise_schedules_lbl = cast_tuple(noise_schedules_lbl)
        noise_schedules_lbl = pad_tuple_to_length(noise_schedules_lbl, 2, 'cosine_p')
        noise_schedules_lbl = pad_tuple_to_length(noise_schedules_lbl, num_unets, 'linear')

        # construct noise schedulers

        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        noise_scheduler_lbl_klass = MultinomialDiffusion
        self.noise_schedulers = nn.ModuleList([])
        self.noise_schedulers_lbl = nn.ModuleList([])

        for timestep, noise_schedule, noise_schedule_lbl in zip(timesteps, noise_schedules, noise_schedules_lbl):
            noise_scheduler = noise_scheduler_klass(noise_schedule=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)
            noise_scheduler_lbl = noise_scheduler_lbl_klass(
                num_classes, noise_schedule=noise_schedule_lbl, timesteps=timestep, p=cosine_p_lbl)
            self.noise_schedulers_lbl.append(noise_scheduler_lbl)

        self.noise_schedulers_sample = nn.ModuleList([])
        self.noise_schedulers_lbl_sample = nn.ModuleList([])

        for sample_timestep, noise_schedule, noise_schedule_lbl in zip(sample_timesteps, noise_schedules, noise_schedules_lbl):
            noise_scheduler_sample = noise_scheduler_klass(noise_schedule=noise_schedule, timesteps=sample_timestep)
            self.noise_schedulers_sample.append(noise_scheduler_sample)
            noise_scheduler_lbl_sample = noise_scheduler_lbl_klass(
                num_classes, noise_schedule=noise_schedule_lbl, timesteps=sample_timestep, p=cosine_p_lbl)
            self.noise_schedulers_lbl_sample.append(noise_scheduler_lbl_sample)

        # randomly cropping for upsampler training

        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert all(map(lambda x: x is None or (isinstance(x, (tuple, list)) and len(x) == 2), self.random_crop_sizes))
        assert not exists(first(self.random_crop_sizes)), \
            'you should not need to randomly crop image during training for base unet, only for upsamplers '\
            '- so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # lowres augmentation noise schedule

        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule=lowres_noise_schedule)
        self.lowres_noise_schedule_lbl = MultinomialDiffusion(
            num_classes, noise_schedule=lowres_noise_schedule, p=cosine_p_lbl)

        # ddpm objectives - predicting noise by default

        self.pred_objectives = cast_tuple(pred_objectives, num_unets)

        # get text encoder

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        self.encode_text = partial(t5_encode_text, name=text_encoder_name)

        # construct unets

        self.unets = nn.ModuleList([])

        self.unet_being_trained_index = -1  # keeps track of which unet is being trained at the moment
        self.only_train_unet_number = only_train_unet_number

        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, (JointUnet, Unet3D, NullUnet))
            is_first = ind == 0

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                cond_on_text=self.condition_on_text,
                text_embed_dim=self.text_embed_dim if self.condition_on_text else None,
                channels=self.channels,
                channels_out=self.channels
            )

            self.unets.append(one_unet)

        # unet image sizes

        self.image_sizes = cast_tuple(image_sizes)
        assert all(map(lambda x: isinstance(x, (tuple, list)) and len(x) == 2, self.image_sizes))

        assert num_unets == len(self.image_sizes), \
            f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {self.image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        # determine whether we are training on images or video

        is_video = any([isinstance(unet, Unet3D) for unet in self.unets])
        self.is_video = is_video

        self.right_pad_dims_to_datatype = partial(rearrange, pattern=(
            'b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))
        self.resize_to = resize_video_to if is_video else resize_image_to

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), \
            'the first unet must be unconditioned (by low resolution image), ' \
            'and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level
        self.lowres_max_thres = lowres_max_thres

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding

        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # p2 loss weight

        self.p2_loss_weight_k = p2_loss_weight_k
        self.p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        assert all([(gamma_value <= 2) for gamma_value in self.p2_loss_weight_gamma]), \
            'in paper, they noticed any gamma greater than 2 is harmful'

        # one temp parameter for keeping track of device

        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)

        # default to device of unets passed in

        self.to(next(self.unets.parameters()).device)

    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def reset_unets_all_one_device(self, device=None):
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        self.unet_being_trained_index = -1

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # overriding state dict functions

    def state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,
        unet: JointUnet,
        x,
        log_lbl,
        t,
        *,
        noise_scheduler: GaussianDiffusionContinuousTimes,
        noise_scheduler_lbl: MultinomialDiffusion,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        lowres_cond_img=None,
        lowres_cond_lbl=None,
        self_cond=None,
        self_cond_lbl=None,
        lowres_noise_times=None,
        cond_scale=1.,
        model_output=None,
        t_next=None,
        pred_objective='noise',
        dynamic_threshold=True,
    ):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'
        lbl = log_onehot_to_index(log_lbl)
        pred, pred_lbl = default(model_output, lambda: unet.forward_with_cond_scale(
            x, lbl, noise_scheduler.get_condition(t),
            text_embeds=text_embeds, text_mask=text_mask,
            cond_images=cond_images, cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img, lowres_cond_lbl=lowres_cond_lbl,
            self_cond=self_cond, self_cond_lbl=self_cond_lbl,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_noise_times)))
        pred_lbl = F.log_softmax(pred_lbl, dim=1)
        pred_lbl = noise_scheduler_lbl.q_posterior(pred_lbl, log_lbl, t)

        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
            # lbl_start = noise_scheduler_lbl.predict_start_from_noise(log_lbl, t=t, noise=pred_lbl) # TODO ???
        elif pred_objective == 'x_start':
            x_start = pred
            # lbl_start = pred_lbl
        else:
            raise ValueError(f'unknown objective {pred_objective}')
        lbl_start = None

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim=-1
            )

            s.clamp_(min=1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        mean_and_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)
        log_lbl = noise_scheduler_lbl.log_sample_categorical(pred_lbl)
        return mean_and_variance, log_lbl, x_start, lbl_start

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        log_lbl,
        t,
        *,
        noise_scheduler,
        noise_scheduler_lbl,
        t_next=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        cond_scale=1.,
        self_cond=None,
        self_cond_lbl=None,
        lowres_cond_img=None,
        lowres_cond_lbl=None,
        lowres_noise_times=None,
        pred_objective='noise',
        dynamic_threshold=True,
    ):
        b, *_, device = *x.shape, x.device
        (model_mean, _, model_log_variance), pred_lbl, x_start, lbl_start = self.p_mean_variance(
            unet, x=x, log_lbl=log_lbl, t=t, t_next=t_next,
            noise_scheduler=noise_scheduler, noise_scheduler_lbl=noise_scheduler_lbl,
            text_embeds=text_embeds, text_mask=text_mask,
            cond_images=cond_images, cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img, lowres_cond_lbl=lowres_cond_lbl,
            self_cond=self_cond, self_cond_lbl=self_cond_lbl,
            lowres_noise_times=lowres_noise_times,
            pred_objective=pred_objective, dynamic_threshold=dynamic_threshold)
        noise = torch.randn_like(x)
        # no noise when t == 0
        is_last_sampling_timestep = (t_next == 0) if isinstance(
            noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, pred_lbl, x_start, lbl_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler: GaussianDiffusionContinuousTimes,
        noise_scheduler_lbl: MultinomialDiffusion,
        lowres_cond_img=None,
        lowres_cond_lbl=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        inpaint_images=None,
        inpaint_labels=None,
        inpaint_masks=None,
        inpaint_resample_times=5,
        init_images=None,
        init_labels=None,
        skip_steps=None,
        cond_scale=1,
        pred_objective='noise',
        dynamic_threshold=True,
        use_tqdm=True
    ):
        assert init_labels is None, 'not implemented yet'
        device = self.device

        batch, _, h, w = shape
        img = torch.randn(shape, device=device)
        uniform_logits = torch.zeros((batch, self.num_classes) + (h, w), device=device)
        log_lbl = noise_scheduler_lbl.log_sample_categorical(uniform_logits)

        # for initialization with an image or video

        if exists(init_images):
            img += init_images
            # TODO init_labels

        # keep track of x0, for self conditioning

        x_start = None
        lbl_start = None

        # prepare inpainting

        has_inpainting = exists(inpaint_images) and exists(inpaint_labels) and exists(inpaint_masks)
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            assert inpaint_masks.shape[1] == 2, \
                f'inpaint mask is a tuple of (mask_image, mask_label) but now:\n{inpaint_labels}'
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-2:])

            log_inpaint_labels = index_to_log_onehot(inpaint_labels.long(), self.num_classes)
            log_inpaint_labels = self.resize_to(log_inpaint_labels, shape[-2:])

            inpaint_masks_image = self.resize_to(inpaint_masks[:, [0]], shape[-2:]).bool()
            inpaint_masks_label = self.resize_to(inpaint_masks[:, [1]], shape[-2:]).bool()

        # time

        timesteps = noise_scheduler.get_sampling_timesteps(batch, device=device)
        timesteps = [t * (t < 1.) + (1 - 1e-7) * (t >= 1.) for t in timesteps]

        # whether to skip any steps

        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        for times, times_next in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps), disable=not use_tqdm):
            is_last_timestep = times_next == 0

            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                if has_inpainting:
                    noised_inpaint_images, _ = noise_scheduler.q_sample(inpaint_images, t=times)
                    img = img * ~inpaint_masks_image + noised_inpaint_images * inpaint_masks_image
                    log_noised_inpaint_labels = noise_scheduler_lbl.q_sample(log_inpaint_labels, t=times)
                    log_lbl = log_lbl * ~inpaint_masks_label + log_noised_inpaint_labels * inpaint_masks_label

                self_cond = x_start if unet.self_cond else None
                self_cond_lbl = lbl_start if unet.self_cond else None

                img, log_lbl, x_start, lbl_start = self.p_sample(
                    unet,
                    img,
                    log_lbl,
                    times,
                    t_next=times_next,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    cond_images=cond_images,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    self_cond_lbl=self_cond_lbl,
                    lowres_cond_img=lowres_cond_img,
                    lowres_cond_lbl=lowres_cond_lbl,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    noise_scheduler_lbl=noise_scheduler_lbl,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold,
                )

                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)
                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )
                    renoised_log_lbl = noise_scheduler_lbl.q_sample_from_to(log_lbl, times_next, times)
                    log_lbl = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        log_lbl,
                        renoised_log_lbl
                    )

        img.clamp_(-1., 1.)

        # final inpainting

        if has_inpainting:
            img = img * ~inpaint_masks_image + inpaint_images * inpaint_masks_image
            log_lbl = log_lbl * ~inpaint_masks_label + log_inpaint_labels * inpaint_masks_label

        unnormalize_img = self.unnormalize_img(img)
        lbl = log_onehot_to_index(log_lbl)
        return unnormalize_img, lbl

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        texts: List[str] = None,
        text_masks=None,
        text_embeds=None,
        video_frames=None,
        cond_images=None,
        inpaint_images=None,
        inpaint_labels=None,
        inpaint_masks=None,
        inpaint_resample_times=5,
        init_images=None,
        init_labels=None,
        skip_steps=None,
        batch_size=1,
        cond_scale=1.,
        lowres_sample_noise_level=None,
        start_at_unet_number=1,
        start_image_or_video=None,
        start_label_or_video=None,
        stop_at_unet_number=None,
        return_all_unet_outputs=False,
        return_pil_images=False,
        device=None,
        use_tqdm=True
    ):
        device = default(device, self.device)
        self.reset_unets_all_one_device(device=device)

        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'

            with autocast(enabled=False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask=True)

            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        if not self.unconditional:
            assert exists(text_embeds), \
                'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim=-1))
            batch_size = text_embeds.shape[0]

        if exists(inpaint_images) and exists(inpaint_labels):
            if self.unconditional:
                if batch_size == 1:  # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            assert inpaint_images.shape[0] == batch_size, \
                'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert inpaint_labels.shape[0] == batch_size, \
                'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            assert not (self.condition_on_text and inpaint_images.shape[0] != text_embeds.shape[0]), \
                'number of inpainting images must be equal to the number of text to be conditioned on'
            assert not (self.condition_on_text and inpaint_labels.shape[0] != text_embeds.shape[0]), \
                'number of inpainting images must be equal to the number of text to be conditioned on'

        assert not (self.condition_on_text and not exists(text_embeds)), \
            'text or text encodings must be passed into imagen if specified'
        assert not (not self.condition_on_text and exists(text_embeds)), \
            'imagen specified not to be conditioned on text, yet it is presented'
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), \
            f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        assert (not (exists(inpaint_images) or exists(inpaint_labels) or exists(inpaint_masks))) \
            or (exists(inpaint_images) and exists(inpaint_labels) and exists(inpaint_masks)), \
            'inpaint images, labels and masks must be both passed in to do inpainting'

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        num_unets = len(self.unets)

        # condition scaling

        cond_scale = cast_tuple(cond_scale, num_unets)

        # add frame dimension for video

        assert not (self.is_video and not exists(video_frames)
                    ), 'video_frames must be passed in on sample time if training on video'

        frame_dims = (video_frames,) if self.is_video else tuple()

        # for initial image and skipping steps

        init_images = cast_tuple(init_images, num_unets)
        init_images = [maybe(self.normalize_img)(init_image) for init_image in init_images]
        init_labels = cast_tuple(init_labels, num_unets)

        skip_steps = cast_tuple(skip_steps, num_unets)

        # handle starting at a unet greater than 1, for training only-upscaler training

        if start_at_unet_number > 1:
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'
            assert exists(start_label_or_video), 'starting image or video must be supplied if only doing upscaling'

            prev_image_size = self.image_sizes[start_at_unet_number - 2]
            img = self.resize_to(start_image_or_video, prev_image_size)
            lbl = self.resize_to(start_label_or_video, prev_image_size)

        # go through each unet in cascade

        for unet_number, unet, channel, image_size, noise_scheduler, noise_scheduler_lbl, pred_objective, \
                dynamic_threshold, unet_cond_scale, unet_init_images, unet_init_labels, unet_skip_steps \
        in tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes,
                    self.noise_schedulers_sample, self.noise_schedulers_lbl_sample, self.pred_objectives,
                    self.dynamic_thresholding, cond_scale, init_images, init_labels, skip_steps),
                    disable=not use_tqdm):

            if unet_number < start_at_unet_number:
                continue

            assert not isinstance(unet, NullUnet), 'one cannot sample from null / placeholder unets'

            context = self.one_unet_in_gpu(unet=unet) if is_cuda else nullcontext()

            with context:
                lowres_cond_img = lowres_cond_lbl = lowres_noise_times = None
                shape = (batch_size, channel, *frame_dims, *image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(
                        batch_size, lowres_sample_noise_level, device=device)

                    lowres_cond_img = self.resize_to(img, image_size)
                    lowres_cond_lbl = self.resize_to(lbl.float(), image_size)

                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(
                        x_start=lowres_cond_img, t=lowres_noise_times, noise=torch.randn_like(lowres_cond_img))
                    lowres_cond_log_lbl = index_to_log_onehot(lowres_cond_lbl.long(), self.num_classes)
                    lowres_cond_log_lbl_noisy = self.lowres_noise_schedule_lbl.q_sample(
                        lowres_cond_log_lbl, t=lowres_noise_times)
                    lowres_cond_lbl_noisy = log_onehot_to_index(lowres_cond_log_lbl_noisy)
                    lowres_cond_lbl = lowres_cond_lbl_noisy  # change just naming

                if exists(unet_init_images) and exists(unet_init_labels):
                    unet_init_images = self.resize_to(unet_init_images, image_size)
                    unet_init_labels = self.resize_to(unet_init_labels, image_size)

                shape = (batch_size, self.channels, *frame_dims, *image_size)

                img, lbl = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    text_mask=text_masks,
                    cond_images=cond_images,
                    inpaint_images=inpaint_images,
                    inpaint_labels=inpaint_labels,
                    inpaint_masks=inpaint_masks,
                    inpaint_resample_times=inpaint_resample_times,
                    init_images=unet_init_images,
                    init_labels=unet_init_labels,
                    skip_steps=unet_skip_steps,
                    cond_scale=unet_cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_cond_lbl=lowres_cond_lbl,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                    noise_scheduler_lbl=noise_scheduler_lbl,
                    pred_objective=pred_objective,
                    dynamic_threshold=dynamic_threshold,
                    use_tqdm=use_tqdm
                )

                outputs.append((img.cpu(), lbl.cpu()))

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        # either return last unet output or all unet outputs
        output_index = -1 if not return_all_unet_outputs else slice(None)

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        assert not self.is_video, 'converting sampled video tensor to video file is not supported yet'

        # TODO lbl pil_images
        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim=0))), outputs))

        # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)
        return pil_images[output_index]

    def p_losses(
        self,
        unet: Union[JointUnet, Unet3D, NullUnet, DistributedDataParallel],
        x_start,
        lbl_start,
        times,
        *,
        noise_scheduler: GaussianDiffusionContinuousTimes,
        noise_scheduler_lbl: MultinomialDiffusion,
        lowres_cond_img=None,
        lowres_cond_lbl=None,
        lowres_aug_times=None,
        text_embeds=None,
        text_mask=None,
        cond_images=None,
        noise=None,
        noise_lbl=None,
        times_next=None,
        pred_objective='noise',
        p2_loss_weight_gamma=0.,
        random_crop_size=None
    ):
        is_video = x_start.ndim == 5

        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise_lbl = default(noise_lbl, lambda: torch.randn_like(x_start))  # TODO

        # normalize to [-1, 1]

        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers

        if exists(random_crop_size):
            if is_video:
                frames = x_start.shape[2]
                x_start, lowres_cond_img, noise = rearrange_many(
                    (x_start, lowres_cond_img, noise), 'b c f h w -> (b f) c h w')

            aug = K.RandomCrop(random_crop_size, p=1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            x_start = aug(x_start)
            lbl_start = aug(lbl_start, params=aug._params)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)
            lowres_cond_lbl = aug(lowres_cond_lbl, params=aug._params)
            noise = aug(noise, params=aug._params)

            if is_video:
                x_start, lowres_cond_img, noise = rearrange_many(
                    (x_start, lowres_cond_img, noise), '(b f) c h w -> b c f h w', f=frames)

        # get x_t

        x_noisy, log_snr = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)
        log_lbl_start = index_to_log_onehot(lbl_start.long(), self.num_classes)
        log_lbl_noisy = noise_scheduler_lbl.q_sample(log_lbl_start, t=times)
        lbl_noisy = log_onehot_to_index(log_lbl_noisy)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3

        lowres_cond_img_noisy = None
        lowres_cond_lbl_noisy = None
        if exists(lowres_cond_img) and exists(lowres_cond_lbl):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(
                x_start=lowres_cond_img, t=lowres_aug_times, noise=torch.randn_like(lowres_cond_img))
            lowres_cond_log_lbl = index_to_log_onehot(lowres_cond_lbl.long(), self.num_classes)
            lowres_cond_log_lbl_noisy = self.lowres_noise_schedule_lbl.q_sample(
                lowres_cond_log_lbl, t=lowres_aug_times)
            lowres_cond_lbl_noisy = log_onehot_to_index(lowres_cond_log_lbl_noisy)

        # time condition

        noise_cond = noise_scheduler.get_condition(times)

        # unet kwargs

        unet_kwargs = dict(
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_images=cond_images,
            lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img=lowres_cond_img_noisy,
            lowres_cond_lbl=lowres_cond_lbl_noisy,
            cond_drop_prob=self.cond_drop_prob,
        )

        # self condition if needed

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond

        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred, pred_lbl = unet.forward(
                    x_noisy,
                    lbl_noisy,
                    noise_cond,
                    **unet_kwargs
                ).detach()
                pred_lbl = F.log_softmax(pred_lbl, dim=1)
                pred_lbl = noise_scheduler_lbl.q_posterior(pred_lbl, log_lbl_noisy, times)

                x_start = noise_scheduler.predict_start_from_noise(
                    x_noisy, t=times, noise=pred) if pred_objective == 'noise' else pred
                # lbl_start = noise_scheduler_lbl.predict_start_from_noise(
                #     lbl_noisy, t=times, noise=pred_lbl) if pred_objective == 'noise' else pred_lbl # TODO ???
                lbl_start = None

                unet_kwargs = {**unet_kwargs, 'self_cond': x_start, 'self_cond_lbl': lbl_start}

        # get prediction

        pred, pred_lbl = unet.forward(
            x_noisy,
            lbl_noisy,
            noise_cond,
            **unet_kwargs
        )
        pred_lbl = F.log_softmax(pred_lbl, dim=1)
        pred_lbl_post = noise_scheduler_lbl.q_posterior(pred_lbl, log_lbl_noisy, times)

        # prediction objective

        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')
        target_log_lbl = noise_scheduler_lbl.q_posterior(log_lbl_start, log_lbl_noisy, times)

        # losses

        losses = self.loss_fn(pred, target, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses_lbl = noise_scheduler_lbl.loss_fn(target_log_lbl, pred_lbl_post, times, log_lbl_start)

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight
            losses_lbl = losses_lbl * loss_weight

        return losses.mean(), losses_lbl.mean()

    def forward(
        self,
        images,
        labels,
        unet: Union[JointUnet, Unet3D, NullUnet, DistributedDataParallel] = None,
        texts: List[str] = None,
        text_embeds=None,
        text_masks=None,
        unet_number=None,
        cond_images=None
    ):
        # assert images.shape[-1] == images.shape[-2], \
        #     f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        assert not (len(self.unets) > 1 and not exists(unet_number)), \
            f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, \
            'you can only train on unet #{self.only_train_unet_number}'

        images = cast_uint8_images_to_float(images)
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        assert is_float_dtype(images.dtype), f'images tensor needs to be floats but {images.dtype} dtype found instead'

        unet_index = unet_number - 1

        unet = default(unet, lambda: self.get_unet(unet_number))

        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        noise_scheduler = self.noise_schedulers[unet_index]
        noise_scheduler_lbl = self.noise_schedulers_lbl[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective = self.pred_objectives[unet_index]
        target_image_size = self.image_sizes[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5

        check_shape(images, 'b c ...', c=self.channels)
        assert h >= target_image_size[0] and w >= target_image_size[1]

        frames = images.shape[2] if is_video else None

        times = noise_scheduler.sample_random_times(b, device=device)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            assert all([*map(len, texts)]), 'text cannot be empty'
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            with autocast(enabled=False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask=True)

            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim=-1))

        assert not (self.condition_on_text and not exists(text_embeds)
                    ), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text and exists(text_embeds)
                    ), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(
            text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        lowres_cond_img = lowres_cond_lbl = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(images, prev_image_size, clamp_range=self.input_image_range)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, clamp_range=self.input_image_range)
            lowres_cond_lbl = self.resize_to(labels, prev_image_size, clamp_range=None)
            lowres_cond_lbl = self.resize_to(lowres_cond_lbl, target_image_size, clamp_range=None)

            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(
                    b, self.lowres_max_thres, device=device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(
                    1, self.lowres_max_thres, device=device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)

        images = self.resize_to(images, target_image_size)
        labels = self.resize_to(labels, target_image_size)

        return self.p_losses(unet, images, labels, times, text_embeds=text_embeds, text_mask=text_masks, cond_images=cond_images, noise_scheduler=noise_scheduler, noise_scheduler_lbl=noise_scheduler_lbl, lowres_cond_img=lowres_cond_img, lowres_cond_lbl=lowres_cond_lbl, lowres_aug_times=lowres_aug_times, pred_objective=pred_objective, p2_loss_weight_gamma=p2_loss_weight_gamma, random_crop_size=random_crop_size)
