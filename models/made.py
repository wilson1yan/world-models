import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('mask', torch.zeros_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.FloatTensor(mask.astype('uint8')))

class MADE(nn.Module):

    def __init__(self, n, hidden_sizes, random_ordering=True):
        super(MADE, self).__init__()
        self.linears = nn.ModuleList()
        output_dim = n

        self.hidden_sizes = hidden_sizes
        self.layer_sizes = hidden_sizes + [output_dim]

        prev_h = n
        for h in self.hidden_sizes:
            self.linears.append(MaskLinear(prev_h, h))
            prev_h = h
        self.out_mu = MaskLinear(prev_h, output_dim)
        self.out_sigma = MaskLinear(prev_h, output_dim)

        self.L = len(hidden_sizes) + 1
        self.n = n

        self.create_masks(random_ordering=random_ordering)

        for mask, linear in zip(self.masks, self.linears):
            linear.set_mask(mask)

        self.out_mu.set_mask(self.mask_out)
        self.out_sigma.set_mask(self.mask_out)

        self.ordering = self.m[0]

    def create_masks(self, random_ordering):
        self.m = []
        if random_ordering:
            self.m.append(np.random.permutation(np.arange(self.n)))
        else:
            self.m.append(np.arange(self.n))
        for l, h in zip(range(1, self.L), self.hidden_sizes):
            min_k = np.min(self.m[l - 1])
            self.m.append(np.random.choice(np.arange(min_k, self.n-1), size=h))
        self.m.append(self.m[0])

        self.masks = [self.m[l][:, np.newaxis] >= self.m[l-1][np.newaxis, :]
                      for l in range(1, self.L)]
        self.mask_out = self.m[self.L][:, np.newaxis] > self.m[self.L-1][np.newaxis, :]

    def forward(self, x):
        for linear in self.linears:
            x = F.relu(linear(x))
        mu = self.out_mu(x)
        logsigma = self.out_sigma(x)
        return mu, logsigma

    def inverse(self, eps, device):
        z = torch.zeros_like(eps).to(device)
        for i in range(self.n):
            out = z
            for linear in self.linears:
                out = F.relu(linear(out))
            mu = self.out_mu(out)
            logsigma = self.out_sigma(out)
            z[:, i] = eps[:, i] * logsigma[:, i].exp() + mu[:, i]
        return z
