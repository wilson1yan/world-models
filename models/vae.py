
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.pixel_cnn.models as models
from models.made import MADE
from models.spatial_softmax import SpatialSoftmax

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_size, latent_size, out_channels=None):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_size[0]

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        if out_channels is None:
            self.deconv4 = nn.ConvTranspose2d(32, img_size[0], 6, stride=2)
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

class Identity(nn.Module):

    def forward(self, x):
        return x

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_size, latent_size, cond_size=None):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_size[0]
        self.cond_size = cond_size

        self.conv1 = nn.Conv2d(img_size[0], 128, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        dim_size = 2*2*256
        if cond_size is not None:
            dim_size += cond_size
        self.fc_mu = nn.Linear(dim_size, latent_size)
        self.fc_logsigma = nn.Linear(dim_size, latent_size)


    def forward(self, x, cond=None): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)

        if self.cond_size is not None:
            assert cond is not None
            x = torch.cat((x, cond), 1)

        mu = self.fc_mu(x)
        logsigma = torch.tanh(self.fc_logsigma(x))

        return mu, logsigma

class SpatialSoftmaxEncoder(nn.Module):
    def __init__(self, img_size, latent_size):
        super(SpatialSoftmaxEncoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_size[0]

        self.conv1 = nn.Conv2d(img_size[0], 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.ssm = SpatialSoftmax(64, 64)

        self.fc_mu = nn.Linear(2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.ssm(x)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_size, latent_size, cond_size=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_size, latent_size, cond_size=cond_size)
        self.decoder = Decoder(img_size, latent_size)

    def sample(self, z, device):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = F.sigmoid(self.decoder(z))
        return recon_x, mu, logsigma, z

    def encode(self, x, cond=None):
        return self.encoder(x, cond=cond)

    def decode_train(self, x, z):
        return F.sigmoid(self.decoder(z))

    def to_device_encoder_only(self, device):
        self.encoder = self.encoder.to(device)

class PixelVAE(nn.Module):
    def __init__(self, img_size, latent_size, n_color_dims,
                 upsample=False, cond_size=None):
        super(PixelVAE, self).__init__()
        self.img_size = img_size
        self.encoder = Encoder(img_size, latent_size, cond_size=cond_size)
        if upsample:
            self.decoder = Decoder(img_size[0], latent_size, out_channels=64)
            self.pixel_cnn = models.LGated(img_size,
                                           64, 120,
                                           n_color_dims=n_color_dims,
                                           num_layers=2,
                                           k=7, padding=3)
        else:
            self.decoder = Identity()
            self.pixel_cnn = models.CGated(img_size,
                                           (latent_size,),
                                           60, num_layers=2,
                                           n_color_dims=n_color_dims,
                                           k=5, padding=2)

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
        return recon_x, mu, logsigma, z

    def encode(self, x, cond=None):
        return self.encoder(x, cond=cond)

    def decode_train(self, x, z):
        z = self.decoder(z)
        return self.pixel_cnn(x, z)

    def to_device_encoder_only(self, device):
        self.encoder = self.encoder.to(device)

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

class VQVAE(nn.Module):
    def __init__(self, img_size, latent_size, latent_dim):
        super(VQVAE, self).__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.latent_dim = latent_dim
