from torch import nn
import tqdm

from models.pixel_cnn.layers import *

class Gated(nn.Module):
    """
    Model combining several gated pixelCNN layers
    """

    def __init__(self, input_size, channels, num_layers,
                 n_color_dims, k=7, padding=3):
        super().__init__()

        c, h, w = input_size
        self.n_color_dims = n_color_dims
        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                MaskedConv2d(
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, self.n_color_dims*c, 1, groups=c)

    def forward(self, x):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xh, xv = x, x

        for layer in self.gated_layers:
            xv, xh = layer((xv, xh))

        x = self.conv2(xh)

        return x.view(b, c, self.n_color_dims, h, w).transpose(1, 2)

    def sample(self, n, img_size, device):
        c, h, w = img_size
        samples = torch.zeros((n,) + img_size).to(device)
        for i in tqdm.trange(h):
            for j in range(w):
                for channel in range(c):

                    result = self(samples)
                    probs = F.softmax(result[:, :, channel, i, j], 1).data

                    pixel_sample = torch.multinomial(probs, 1).float() / (self.n_color_dims - 1)
                    samples[:, channel, i, j] = pixel_sample.squeeze()
        return samples


class LGated(nn.Module):
    """
    Gated model with location specific conditional
    """

    def __init__(self, input_size, conditional_channels, channels, num_layers,
                 n_color_dims, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.n_color_dims = n_color_dims
        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                LMaskedConv2d(
                    (channels, h, w),
                    conditional_channels,
                    channels, colors=c, self_connection=i > 0,
                    res_connection=i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, n_color_dims*c, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)
        return x.view(b, c, self.n_color_dims, h, w).transpose(1, 2)

    def sample(self, img_size, device, cond):
        c, h, w = img_size
        samples = torch.zeros((cond.size(0),) + img_size).to(device)
        for i in tqdm.trange(h):
            for j in range(w):
                for channel in range(c):

                    result = self(samples, cond)
                    probs = F.softmax(result[:, :, channel, i, j], 1).data

                    pixel_sample = torch.multinomial(probs, 1).float() / (self.n_color_dims - 1)
                    samples[:, channel, i, j] = pixel_sample.squeeze()
        return samples

class CGated(nn.Module):
    """
    Gated model with location-independent conditional
    """

    def __init__(self, input_size, cond_size, channels, num_layers,
                 n_color_dims, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.n_color_dims = n_color_dims
        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                CMaskedConv2d(
                    (channels, h, w),
                    cond_size,
                    channels, colors=c, self_connection=i > 0,
                    res_connection=i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, self.n_color_dims*c, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xv, xh = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x.view(b, c, self.n_color_dims, h, w).transpose(1, 2)

    def sample(self, img_size, device, cond):
        c, h, w = img_size
        samples = torch.zeros((cond.size(0),) + img_size).to(device)
        for i in tqdm.trange(h):
            for j in range(w):
                for channel in range(c):

                    result = self(samples, cond)
                    probs = F.softmax(result[:, :, channel, i, j], 1).data

                    pixel_sample = torch.multinomial(probs, 1).float() / (self.n_color_dims - 1)
                    samples[:, channel, i, j] = pixel_sample.squeeze()
        return samples
