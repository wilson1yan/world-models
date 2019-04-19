
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.pixel_cnn.models as models
from models.made import MADE

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size, out_channels=None):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        if out_channels is None:
            self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)
        else:
            self.deconv4 = nn.ConvTranspose2d(32, out_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = self.deconv4(x)
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def sample(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = F.sigmoid(self.decoder(z))
        return recon_x, mu, logsigma

class PixelVAE(nn.Module):
    def __init__(self, img_size, latent_size, n_color_dims,
                 upsample=False):
        super(PixelVAE, self).__init__()
        self.img_size = img_size
        self.encoder = Encoder(img_size[0], latent_size)
        if upsample:
            self.decoder = Decoder(img_size[0], latent_size, out_channels=64)
            self.pixel_cnn = models.LGated(img_size,
                                           64, 120,
                                           n_color_dims=n_color_dims,
                                           num_layers=4,
                                           k=7, padding=3)
        else:
            self.decoder = lambda x: x
            self.pixel_cnn = models.CGated(img_size,
                                           (latent_size,),
                                           120, num_layers=4,
                                           n_color_dims=n_color_dims,
                                           k=7, padding=3)

    def sample(self, z, device):
        z = self.decoder(z)
        return self.pixel_cnn.sample(self.img_size, device,
                                     cond=z)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        z = self.decoder(z)
        recon_x = self.pixel_cnn(x, z)
        return recon_x, mu, logsigma

class AFPixelVAE(nn.Module):
    def __init__(self, img_size, latent_size, n_color_dims,
                 upsample=False):
        super(AFPixelVAE, self).__init__()
        self.img_size = img_size
        self.prior = MADE(latent_size, [256, 256, 256, 256])
        self.encoder = Encoder(img_size[0], latent_size)
        if upsample:
            self.decoder = Decoder(img_size[0], latent_size, out_channels=64)
            self.pixel_cnn = models.LGated(img_size,
                                           64, 120,
                                           n_color_dims=n_color_dims,
                                           num_layers=4,
                                           k=7, padding=3)
        else:
            self.decoder = lambda x: x
            self.pixel_cnn = models.CGated(img_size,
                                           (latent_size,),
                                           120, num_layers=4,
                                           n_color_dims=n_color_dims,
                                           k=7, padding=3)

    def sample(self, eps, device):
        z = self.prior.inverse(eps, device)
        z = self.decoder(z)
        return self.pixel_cnn.sample(self.img_size, device,
                                     cond=z)

    def forward(self, x):
        mu_z, logsigma_z = self.encoder(x)
        sigma_z = logsigma_z.exp()
        eps = torch.randn_like(sigma_z)
        z = eps.mul(sigma_z).add_(mu_z)

        mu_eps, logsigma_eps = self.prior(z)
        eps = (z - mu_eps) * torch.exp(-logsigma_eps)
        logdet = -torch.sum(logsigma_eps.view(z.size(0), -1), -1)

        cond = self.decoder(z)
        recon_x = self.pixel_cnn(x, cond)
        return recon_x, mu_z, logsigma_z, eps, logdet, z
