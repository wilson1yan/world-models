import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, h, w, temperature=1, clim=1):
        super(SpatialSoftmax, self).__init__()
        
        hvals = torch.linspace(-clim, clim, h).view(1, 1, h, 1)
        wvals = torch.linspace(-clim, clim, w).view(1, 1, 1, w)
        self.register_buffer('hvals', hvals)
        self.register_buffer('wvals', wvals)

        self.temperature = temperature

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, h * w)
        x = F.softmax(x / self.temperature, 1)
        x = x.view(b, c, h, w)

        avg_h = (x * self.hvals).sum(-1).sum(-1)
        avg_w = (x * self.wvals).sum(-1).sum(-1)

        return torch.cat([avg_h, avg_w], 1)
