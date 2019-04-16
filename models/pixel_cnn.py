import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda')

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'

        super(MaskConv2d, self).__init__(*args, **kwargs)
        self.mask = self.create_mask(mask_type).cuda(device)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        mask = torch.zeros_like(self.weight)
        mask[:, :, :k // 2] = 1
        mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            mask[:, :, k // 2, k // 2] = 1
        return mask

class PixelCNN(nn.Module):
    def __init__(self, img_channels, n_color_dim, n_blocks=5, cond=True):
        super(PixelCNN, self).__init__()
        if cond:
            conv = MaskConv2d('A', img_channels, 128 - img_channels, 7, padding=3)
        else:
            conv = MaskConv2d('A', img_channels, 128, 7, padding=3)
        residual_blocks = nn.ModuleList([self.build_residual_block()
                                         for _ in range(n_blocks)])
        self.first_layer = nn.Sequential(conv, nn.ReLU())
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.out = nn.Sequential(nn.Conv2d(128, 256, 1), nn.ReLU(),
                                 nn.Conv2d(256, img_channels * n_color_dim, 1))
        self._image_shape = (img_channels, 64, 64)
        self.n_color_dim = n_color_dim

    def sample(self, cond=None):
        if cond is None:
            images = torch.zeros((64,) + self._image_shape).cuda()
        else:
            images = torch.zeros_like(cond)

        for r in range(images.size(2)):
            for c in range(images.size(3)):
                out = self(images, cond=cond)
                for channel in range(3):
                    probs = F.softmax(out[:, :, channel, r, c], 1).data
                    pixel_sample = torch.multinomial(probs, 1).float() / (self.n_color_dim - 1)
                    images[:, channel, r, c] = pixel_sample.squeeze(1)
        return images

    def forward(self, x, cond=None):
        input_size = x.size()[1:]

        x = self.first_layer(x)
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        for block in self.residual_blocks:
            x = block(x) + x
        x = self.out(x)
        x = x.view(x.size(0), self.n_color_dim, *input_size)
        return x

    def build_residual_block(self):
        return nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            MaskConv2d('B', 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.ReLU()
        )
