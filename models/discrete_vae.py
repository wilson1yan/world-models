import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import ResBlock

class DiscreteVAE(nn.Module):

    def __init__(self, img_size, dim):
        super(DiscreteVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_size[0], dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            # nn.Conv2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            # nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            nn.ConvTranspose2d(dim, img_size[0], 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        pass
