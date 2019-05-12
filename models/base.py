import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[],
                 output_activation=None):
        super(MLP, self).__init__()

        modules = []
        hprev = input_dim
        for h in hidden_dims + [output_dim]:
            modules.append(nn.Linear(hprev, h))
            modules.append(nn.ReLU())
            hprev = h
        modules.pop(-1)
        self.model = nn.Sequential(*modules)
        self.output_activation = output_activation

    def forward(self, x):
        out = self.model(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

class CondConv2d(nn.Module):

    def __init__(self, input_size, in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, cond_size=None):
        super(CondConv2d, self).__init__()
        self.cond_size = cond_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        c, h, w = input_size
        self.output_shape = (out_channels, int((h-kernel_size+2*padding)/stride + 1),
                             int((w-kernel_size+2*padding)/stride + 1))
        if cond_size is not None:
            to = int(np.prod(self.output_shape))
            self.cond_layer = nn.Linear(cond_size, to)

    def forward(self, x, cond=None):
        if self.cond_size is not None:
            assert cond is not None
        x = self.conv(x)
        if self.cond_size is not None:
            cond = self.cond_layer(cond).view(x.size(0), *self.output_shape)
            x += cond
        return x

class SimpleConv(nn.Module):
    def __init__(self, in_channels, n_filters, reduce_factor=0):
        super(SimpleConv, self).__init__()
        self.n_filters = n_filters

        layers = []
        layers.append(nn.Conv2d(in_channels, n_filters, 3, padding=1))
        layers.append(nn.ReLU())
        c = n_filters
        for _ in range(reduce_factor):
            layers.append(nn.Conv2d(c, n_filters, 3, padding=1, stride=2))
            layers.append(nn.ReLU())
            c = n_filters
        layers.pop()
        self.process = nn.Sequential(*layers)

    def forward(self, x):
        x = self.process(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
