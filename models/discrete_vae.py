import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import ResBlock, CondConv2d

class CondEncoder(nn.Module):

    def __init__(self, img_size, dim, n_categories, cond_size=None):
        super(CondEncoder, self).__init__()
        self.conv1 = CondConv2d(img_size, img_size[0], dim,
                                4, 2, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = CondConv2d((dim, 32, 32), dim, dim,
                                4, 2, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv3 = CondConv2d((dim, 16, 16), dim, dim,
                                4, 2, 1, cond_size=cond_size)
        self.res_blocks = nn.Sequential(ResBlock(dim), ResBlock(dim))
        self.conv4 = CondConv2d((dim, 8, 8), dim, n_categories,
                                3, 1, 1, cond_size=cond_size)

    def forward(self, x, cond=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x, cond=cond)
        x = self.res_blocks(x)
        x = F.relu(x)
        x = self.conv4(x, cond=cond)
        return F.log_softmax(x, 1)

class DiscreteVAE(nn.Module):

    def __init__(self, img_size, dim, n_categories, cond_size=None):
        super(DiscreteVAE, self).__init__()

        self.encoder = CondEncoder(img_size, dim, n_categories,
                                   cond_size=cond_size)
        self.embedding = nn.Linear(n_categories, 128)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, dim, 3, 1, 1),
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            nn.ConvTranspose2d(dim, img_size[0], 4, 2, 1),
            nn.Sigmoid()
        )

        self.cond_size = cond_size
        self.temperature = 1.0
        self.min_temp = 0.5
        self.anneal_rate = 0.0003
        self.t = 0
        self.n_categories = n_categories

    def anneal(self):
        self.temperature = max(self.min_temp, 1.0 * np.exp(-self.anneal_rate * self.t))
        self.t += 1

    def encode(self, x, cond=None):
        log_probs = self.encoder(x, cond=cond)
        return torch.max(log_probs, 1)[1], None

    def encode_train(self, x, cond=None):
        log_probs = self.encoder(x, cond=cond)
        z = self.reparam(log_probs)
        z = z.permute(0, 2, 3, 1).contiguous()
        embeddings = self.embedding(z)
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
        return embeddings, log_probs

    def decode(self, z, device):
        z = self.to_embedding(z, device)
        return self.decoder(z)

    def decode_train(self, embeddings):
        return self.decoder(embeddings)

    def to_embedding(self, latents, device):
        one_hot = torch.zeros(*latents.size(), self.n_categories).to(device)
        one_hot.scatter_(-1, latents.long().unsqueeze(-1), 1)
        z = self.embedding(one_hot)
        z = z.permute(0, 3, 1, 2).contiguous()
        return z

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
        embeddings = self.embedding(z)
        embeddings = embeddings.permute(0, 3, 1, 2)
        recon_x = self.decoder(embeddings)

        return recon_x, log_probs
