import numpy as np
from scipy.stats import ortho_group

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import ASIZE, RED_SIZE

C, H, W = ASIZE, RED_SIZE, RED_SIZE

def torch_mean(x, dims, keepdim=False):
    dims = sorted(dims, reverse=True)
    for d in dims:
        x = x.mean(d, keepdim=keepdim)
    return x


def torch_sum(x, dims, keepdim=False):
    dims = sorted(dims, reverse=True)
    for d in dims:
        x = x.sum(d, keepdim=keepdim)
    return x


def create_mask(type, channels, reduction_id):
    assert type in ['checkerboard', 'channel', 'bad']

    w = h = int(np.sqrt(C * H * W / channels / (2 ** reduction_id)))

    if type == 'checkerboard':
        re = np.r_[w // 2 * [0, 1]]
        ro = np.r_[w // 2 * [1, 0]]
        checkerboard = np.row_stack(h // 2 * (re, ro))
        checkerboard = np.expand_dims(checkerboard, axis=0).repeat(channels, axis=0)
        checkerboard = np.expand_dims(checkerboard, axis=0)
        return checkerboard
    elif type == 'channel':
        ones = np.ones((1, 1, w, h))
        zeros = np.zeros((1, channels - 1, w, h))
        return np.concatenate((ones, zeros), axis=1)
    else:
        ones = np.ones((1, channels, h, w // 2))
        zeros = np.zeros((1, channels, h, w // 2))
        return np.concatenate((ones, zeros), axis=3)


class Preprocess(nn.Module):

    def __init__(self):
        super(Preprocess, self).__init__()
        self.alpha = 0.05

    def forward(self, x, logdet):
        x = self.alpha + (1 - self.alpha) * x / 4
        x = self._logit(x)
        logdet += (-F.logsigmoid(x) - torch.log(1 - torch.sigmoid(x)) + \
                   np.log(1 - self.alpha) - np.log(4)).view(x.size(0), -1).sum(-1)

        return x, logdet

    def inverse(self, y):
        x = (torch.sigmoid(y) - self.alpha) * 4 / (1 - self.alpha)
        return x

    def _logit(self, x):
        return torch.log(x + 1e-7) - torch.log(1 - x + 1e-7)


class InvertibleConv(nn.Module):
    def __init__(self, in_channels):
        super(InvertibleConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, in_channels, 1, 1))

        self.in_channels = in_channels
        self._init_weight()

    def forward(self, x, logdet):
        b, c, h, w = x.size()
        out = F.conv2d(x, self.weight)
        detW = torch.det(self.weight.squeeze(2).squeeze(2))
        logdet += (h * w * torch.log(torch.abs(detW) + 1e-7)).repeat(x.size(0))
        return out, logdet

    def inverse(self, y):
        invW = torch.inverse(self.weight.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        x =  F.conv2d(y, invW)
        return x

    def _init_weight(self):
        rot_matrix = ortho_group.rvs(self.in_channels)
        self.weight.data.copy_(torch.Tensor(rot_matrix).unsqueeze_(2).unsqueeze_(3))


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, init, reduction=0, mask=None):
        super(AffineCoupling, self).__init__()
        w = h = int(np.sqrt(C * W * H / in_channels / (2 ** reduction)))
        self.register_buffer('mask', torch.zeros((1, in_channels, w, h)))
        self.model = init()
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.tanh = nn.Tanh()

        if mask is not None:
            self.set_mask(mask)

    def forward(self, x, logdet):
        y1 = x * self.mask
        y2 = x * (1 - self.mask)
        out_s_t = self.model(y1)
        log_s, t = torch.chunk(out_s_t, 2, dim=1)
        log_s, t = self.scale * self.tanh(log_s) * (1 - self.mask), t * (1 - self.mask)
        logdet += log_s.view(log_s.size(0), -1).sum(-1)
        y = y1 + torch.exp(log_s) * y2 + t
        return y, logdet

    def inverse(self, y):
        x1 = y * self.mask
        out_s_t = self.model(x1)
        log_s, t = torch.chunk(out_s_t, 2, dim=1)
        log_s, t  = self.tanh(log_s) / self.scale * (1 - self.mask), t * (1 - self.mask)
        x = (y - t) * torch.exp(-log_s)
        return x

    def set_mask(self, mask):
        self.mask.data.copy_(torch.FloatTensor(mask.astype('uint8')))


class ActNorm(nn.Module):
    def __init__(self, in_channels, input_dim):
        super(ActNorm, self).__init__()

        assert input_dim == 2 or input_dim == 4
        if input_dim == 2:
            size = [1, in_channels]
        else:
            size = [1, in_channels, 1, 1]
        self.register_parameter('bias', nn.Parameter(torch.zeros(*size)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(*size)))
        self.initialized = False
        self.input_dim = input_dim

    def initialize(self, x):
        with torch.no_grad():
            if self.input_dim == 2:
                bias = torch_mean(x, [0, 1], keepdim=True)
                scale = torch_mean((x - bias) ** 2, [0, 1], keepdim=True)
            else:
                bias = torch_mean(x, [0, 2, 3], keepdim=True)
                scale = torch_mean((x - bias) ** 2, [0, 2, 3], keepdim=True)
            scale = torch.sqrt(scale)
            log_scale = torch.log(1 / (scale + 1e-7))

            self.bias.data.copy_(bias.data)
            self.log_scale.data.copy_(log_scale.data)
            self.initialized = True

    def forward(self, x, logdet):
        assert len(x.size()) == self.input_dim
        if not self.initialized:
            self.initialize(x)

        out = (x - self.bias) * torch.exp(self.log_scale)
        if self.input_dim == 2:
            logdet += (self.log_scale.sum(1) * x.size(1)).repeat(x.size(0))
        else:
            logdet += (self.log_scale.sum() * x.size(2) * x.size(3)).repeat(x.size(0))

        return out, logdet

    def inverse(self, y):
        return y * torch.exp(-self.log_scale) + self.bias


class ChannelSqueeze(nn.Module):

    def __init__(self, reverse=False):
        super(ChannelSqueeze, self).__init__()
        self.reverse = reverse

    def forward(self, x, logdet):
        x = self.unsqueeze(x) if self.reverse else self.squeeze(x)
        return x, logdet

    def inverse(self, y):
        return self.squeeze(y) if self.reverse else self.unsqueeze(y)

    def squeeze(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, h // 2, 2, w // 2, 2, c)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, h // 2, w // 2, c * 4)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def unsqueeze(self, x):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, h, w, c // 4, 2, 2)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, h * 2, w * 2, c // 4)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, 1)

    def forward(self, x):
        original = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x + original
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_filters=128, n_blocks=8):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(n_filters, n_filters)
                                     for _ in range(n_blocks)])
        self.conv2 = nn.Conv2d(n_filters, in_channels * 2, 3, padding=1)
        self.in_channels = in_channels

    def forward(self, x):
        out = F.relu(self.conv1(x))
        for block in self.blocks:
            out = block(out)
        out = self.conv2(out)
        return out

class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()
        self.model = nn.ModuleList()

        self.model.append(Preprocess())
        self.create_affine_block(2, 'checkerboard', 3)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(2, 'channel', 12)
        self.create_affine_block(2, 'checkerboard', 12)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(2, 'channel', 48)
        self.create_affine_block(2, 'checkerboard', 48)

        self.latent_size = None

    def create_affine_block(self, n, type, in_channels):
        mask = create_mask(type, in_channels, 0)
        init = lambda: ResNet(in_channels)
        for i in range(n):
            if i % 2 == 0:
                layer_mask = mask
            else:
                layer_mask = 1 - mask
            self.model.append(ActNorm(in_channels, 4))
            self.model.append(AffineCoupling(in_channels, init,
                                             mask=layer_mask))

    def forward(self, x):
        out, logdet = x, 0
        for layer in self.model:
            out, logdet = layer(out, logdet)

        if self.latent_size is None:
            self.latent_size = out.size()[1:]

        return out, logdet

    def sample(self, z):
        out = z
        for layer in reversed(self.model):
            out = layer.inverse(out)
        return out


class Glow(nn.Module):
    def __init__(self):
        super(Glow, self).__init__()
        self.model = nn.ModuleList()

        self.model.append(Preprocess())
        self.create_affine_block(3, 'channel', 3)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(3, 'channel', 12)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(3, 'channel', 48)

        self.latent_size = None

    def create_affine_block(self, n, type, in_channels):
        mask = create_mask(type, in_channels, 0)
        init = lambda: ResNet(in_channels)
        for _ in range(n):
            self.model.append(ActNorm(in_channels, 4))
            self.model.append(InvertibleConv(in_channels))
            self.model.append(AffineCoupling(in_channels, init,
                                             mask=mask))

    def forward(self, x):
        out, logdet = x, 0
        for layer in self.model:
            out, logdet = layer(out, logdet)

        if self.latent_size is None:
            self.latent_size = out.size()[1:]

        return out, logdet

    def sample(self, z):
        out = z
        for layer in reversed(self.model):
            out = layer.inverse(out)
        return out

class SqueezeSplit(nn.Module):
    def __init__(self, n, channel_size, reduction_id):
        super(SqueezeSplit, self).__init__()
        self.reduction_id = reduction_id
        self.model = nn.ModuleList()
        self.model.append(ChannelSqueeze(reverse=False))
        channel_size *= 4
        self.create_affine_block(n, 'channel', channel_size)

    def create_affine_block(self, n, type, in_channels):
        mask = create_mask(type, in_channels, self.reduction_id)
        init = lambda: ResNet(in_channels)
        for _ in range(n):
            self.model.append(ActNorm(in_channels, 4))
            self.model.append(InvertibleConv(in_channels))
            self.model.append(AffineCoupling(in_channels, init,
                                             reduction=self.reduction_id,
                                             mask=mask))

    def forward(self, x, logdet):
        out = x
        for layer in self.model:
            out, logdet = layer(out, logdet)
        z1, z2 = out.chunk(2, dim=1)
        return z1, z2, logdet

    def inverse(self, z1, z2):
        z = torch.cat((z1, z2), dim=1)
        for layer in reversed(self.model):
            z = layer.inverse(z)
        return z

class MultiscaleGlow(nn.Module):
    def __init__(self, n_scale=3, n_blocks=6):
        super(MultiscaleGlow, self).__init__()
        self.n_scale = n_scale

        self.preprocess = Preprocess()
        self.multiscale_layers = nn.ModuleList()

        channel_size = 3
        for i in range(n_scale):
            self.multiscale_layers.append(SqueezeSplit(n_blocks, channel_size, i))
            channel_size *= 2

        self.output_layers = nn.ModuleList()
        self.output_layers.append(ChannelSqueeze(reverse=False))
        channel_size *= 4
        self.create_affine_block(n_blocks, 'channel', channel_size, self.output_layers)

        self.latent_size = None
        self.multiscale_sizes = None
        self.multiscale_totals = None

    def create_affine_block(self, n, type, in_channels, module_list):
        mask = create_mask(type, in_channels, self.n_scale)
        init = lambda: ResNet(in_channels)
        for _ in range(n):
            module_list.append(ActNorm(in_channels, 4))
            module_list.append(InvertibleConv(in_channels))
            module_list.append(AffineCoupling(in_channels, init,
                                              reduction=self.n_scale,
                                              mask=mask))

    def forward(self, x):
        out, logdet = x, 0
        out, logdet = self.preprocess(out, logdet)

        zs = []
        for multiscale in self.multiscale_layers:
            z, out, logdet = multiscale(out, logdet)
            zs.append(z)

        for output_layer in self.output_layers:
            out, logdet = output_layer(out, logdet)

        zs.append(out)
        final_z = torch.cat([z.view(z.size(0), -1) for z in zs], dim=1)

        if self.latent_size is None:
            self.multiscale_sizes = [z.size()[1:] for z in zs]
            self.multiscale_totals = [np.prod(s) for s in self.multiscale_sizes]
            self.latent_size = final_z.size()[1:]

        return final_z, logdet

    def sample(self, z):
        cum_sum = np.cumsum(self.multiscale_totals)
        zs = []
        lower = 0
        for i in range(self.n_scale + 1):
            zs.append(z[:,lower:cum_sum[i]].view(z.size(0), *self.multiscale_sizes[i]))
            lower = cum_sum[i]

        last_z = zs[-1]
        for output_layer in reversed(self.output_layers):
            last_z = output_layer.inverse(last_z)

        for first_z, layer in zip(reversed(zs[:-1]), reversed(self.multiscale_layers)):
            last_z = layer.inverse(first_z, last_z)

        last_z = self.preprocess.inverse(last_z)
        return last_z
