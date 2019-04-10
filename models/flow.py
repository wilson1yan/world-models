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


def create_mask(type, channels):
    assert type in ['checkerboard', 'channel', 'bad']

    w = h = int(np.sqrt(C * H * W / channels))

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
        x = self.alpha + (1 - self.alpha) * x / 256
        x = self._logit(x)
        logdet += (-F.logsigmoid(x) - torch.log(1 - torch.sigmoid(x)) + \
                   np.log(1 - self.alpha) - np.log(256)).view(x.size(0), -1).sum(-1)

        return x, logdet

    def inverse(self, y):
        x = (torch.sigmoid(y) - self.alpha) * 256 / (1 - self.alpha)
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
    def __init__(self, in_channels, init, mask=None):
        super(AffineCoupling, self).__init__()
        w = h = int(np.sqrt(C * W * H / in_channels))
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

class Glow(nn.Module):
    def __init__(self):
        super(Glow, self).__init__()
        self.model = nn.ModuleList()

        self.model.append(Preprocess())
        self.create_affine_block(2, 'channel', 3)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(2, 'channel', 12)
        self.model.append(ChannelSqueeze(reverse=False))
        self.create_affine_block(2, 'channel', 48)

        self.latent_size = None

    def create_affine_block(self, n, type, in_channels):
        mask = create_mask(type, in_channels)
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
