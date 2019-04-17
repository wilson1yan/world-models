import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda')

class LMaskedConv2d(nn.Module):
    """
    Masked convolution, with location dependent conditional.
    The conditional must be an 'image' tensor (BCHW) with the same
    resolution as the instance (no of channels can be different)
    """
    def __init__(self, input_size, conditional_channels, channels,
                 colors=3, self_connection=False, res_connection=True,
                 hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k,
                                    padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1,
                                    padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1,
                                    padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R,
        # and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors",
            # R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

        # The conditional weights
        self.vhf = nn.Conv2d(conditional_channels, channels, 1)
        self.vhg = nn.Conv2d(conditional_channels, channels, 1)
        self.vvf = nn.Conv2d(conditional_channels, channels, 1)
        self.vvg = nn.Conv2d(conditional_channels, channels, 1)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h,  (self.vvf, self.vvg))
            hx = self.gate(hx, h,  (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation,
        functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.
        Conditional and weights are used to compute a bias based on the conditional element
        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond)
        sig_bias = vg(cond)

        # compute convolution term
        b = x.size(0)
        c = x.size(1)

        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return F.tanh(top + tan_bias) * F.sigmoid(bottom + sig_bias)

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
            conv = MaskConv2d('A', img_channels, 128, 7, padding=3)
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
