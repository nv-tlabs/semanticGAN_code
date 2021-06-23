# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

from torch import nn
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from models.utils import *
import random

class ToSEG(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out



class GeneratorSeg(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            seg_dim=1,
            image_mode='L',
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        self.seg_dim = seg_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        if image_mode == 'RGB':
            self.rgb_channel = 3
        else:
            self.rgb_channel = 1

        self.to_rgb1 = ToRGB(self.channels[4], self.rgb_channel, style_dim, upsample=False)
        self.to_seg1 = ToSEG(self.channels[4], self.seg_dim, style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_segs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            # print(i, in_channel, out_channel)
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, self.rgb_channel, style_dim))
            self.to_segs.append(ToSEG(out_channel, self.seg_dim, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def sample(self, num, latent_space_type='Z'):
        """Samples latent codes randomly.
        Args:
        num: Number of latent codes to sample. Should be positive.
        latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
        Returns:
        A `numpy.ndarray` as sampled latend codes.
        Raises:
        ValueError: If the given `latent_space_type` is not supported.
        """
        latent_space_type = latent_space_type.upper()
        if latent_space_type == 'Z':
            latent_codes = np.random.randn(num, self.style_dim)
        elif latent_space_type == 'W':
            latent_codes = np.random.randn(num, self.style_dim)
        elif latent_space_type == 'WP':
            latent_codes = np.random.randn(num, self.n_latent, self.style_dim)
        else:
            raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

        return latent_codes.astype(np.float32)

    def forward(
            self,
            styles,
            return_latents=False,
            return_latent_output=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            return_mimaps=False,
    ):

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)

        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip_rgb = self.to_rgb1(out, latent[:, 1])
        skip_seg = self.to_seg1(out, latent[:, 1])

        if return_mimaps:
            image_mimaps = []
            label_mimaps = []
            skip_rgb_norm = skip_rgb.detach().cpu()
            skip_rgb_norm = (skip_rgb_norm - skip_rgb_norm.min()) / (skip_rgb_norm.max() - skip_rgb_norm.min())
            skip_rgb_norm = skip_rgb_norm[0].permute(1, 2, 0)

            skip_seg_norm = skip_seg.detach().cpu()
            skip_seg_norm = (skip_seg_norm - skip_seg_norm.min()) / (skip_seg_norm.max() - skip_seg_norm.min())
            skip_seg_norm = skip_seg_norm[0].permute(1, 2, 0)

            image_mimaps.append(skip_rgb_norm)
            label_mimaps.append(skip_seg_norm)

        i = 1

        for conv1, conv2, noise1, noise2, to_rgb, to_seg in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs, self.to_segs
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            out = conv2(out, latent[:, i + 1], noise=noise2)

            skip_rgb = to_rgb(out, latent[:, i + 2], skip_rgb)

            skip_seg = to_seg(out, latent[:, i + 2], skip_seg)

            if return_mimaps:
                skip_rgb_norm = skip_rgb.detach().cpu()
                skip_rgb_norm = (skip_rgb_norm - skip_rgb_norm.min()) / (skip_rgb_norm.max() - skip_rgb_norm.min())
                skip_rgb_norm = skip_rgb_norm[0].permute(1, 2, 0)

                skip_seg_norm = skip_seg.detach().cpu()
                skip_seg_norm = (skip_seg_norm - skip_seg_norm.min()) / (skip_seg_norm.max() - skip_seg_norm.min())
                skip_seg_norm = skip_seg_norm[0].permute(1, 2, 0)

                image_mimaps.append(skip_rgb_norm)
                label_mimaps.append(skip_seg_norm)

            i += 2

        image = skip_rgb
        seg = skip_seg

        if return_latents:
            return image, latent
        elif return_latent_output:
            return image, seg, latent
        elif return_mimaps:
            return image_mimaps, label_mimaps
        else:
            return image, seg


class Discriminator(nn.Module):
    def __init__(self, size, input_dim, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(input_dim, self.channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = self.channels[size]

        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, self.channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(self.channels[4] * 4 * 4, self.channels[4], activation='fused_lrelu'),
            EqualLinear(self.channels[4], 1),
        )

    def _cal_stddev(self, x):
        batch, channel, height, width = x.shape
        group = min(batch, self.stddev_group)
        stddev = x.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        x = torch.cat([x, stddev], 1)

        return x

    def forward(self, input):
        batch = input.shape[0]

        out = self.convs(input)

        out = self._cal_stddev(out)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer_opt='spectralinstance',
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer_opt, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer_opt='spectralinstance', use_sigmoid=False,
                 getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        norm_layer = get_norm_layer(norm_layer_opt)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]

        else:
            return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode

        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


def get_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
