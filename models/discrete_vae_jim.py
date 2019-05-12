import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import ResBlock

class DiscreteVAE(nn.Module):

    def __init__(self, img_size, dim, n_categories):
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
            nn.ReLU(True),
            nn.Conv2d(dim, n_categories, 4, 2, 1),
            nn.BatchNorm2d(n_categories),
            nn.ReLU(True),
            nn.LogSoftmax(1),
        )

        # self.embedding = nn.Linear(n_categories, 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, dim, 4, 2, 1),
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
            nn.BatchNorm2d(img_size[0]),
            # nn.Tanh()
            nn.Sigmoid()
        )

        self.temperature = 1.0
        self.min_temp = 0.5
        self.anneal_rate = 0.00003
        self.t = 0
        self.n_categories = n_categories

    def anneal(self):
        self.temperature = max(self.min_temp, 1.0 * np.exp(-self.anneal_rate * self.t))
        self.t += 1

    def encode(self, x):
        log_probs = self.encoder(x)
        return torch.max(log_probs, 1)[1], None

    def decode(self, z, device):
        one_hot = torch.zeros(*z.size(), self.n_categories).to(device)
        one_hot.scatter_(-1, z.long().unsqueeze(-1), 1)
        z = one_hot
        z = self.embedding(z)
        z = z.permute(0, 3, 1, 2)
        return self.decoder(z)

    def sample(self, z, device):
        return self.decode(z, device)

    def reparam(self, log_probs): # Any shape, log_probs in dimension 1
        eps = torch.rand_like(log_probs)
        gumbel = -torch.log(-torch.log(eps + 1e-8) + 1e-8)
        z = log_probs + gumbel
        return F.softmax(z / self.temperature, 1)


    def forward(self, x):
        log_probs = self.encoder(x) # Outputs B x N_CAT x 16 x 16

        b, c, h, w = log_probs.size()

        z = self.reparam(log_probs)

        z = z.permute(0, 2, 3, 1).contiguous()

        shape = z.size()
        _, ind = z.max(dim=-1)
        z_hard = torch.zeros_like(z).view(-1, shape[-1])

        z_hard.scatter_(1, ind.view(-1, 1), 1)
        z_hard = z_hard.view(*shape)
        
        z_hard = (z_hard - z).detach() + z

        # embeddings = self.embedding(z_hard)

        # embeddings = embeddings.permute(0, 3, 1, 2)

        embeddings = z_hard.permute(0, 3, 1, 2)


        recon_x = self.decoder(embeddings)

        return recon_x, log_probs
