
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters

        self.model = []
        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(in_channels, n_filters, 3, padding=1))
        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(n_filters, n_filters, 1))
        self.model.append(nn.ReLU())
        self.model.append(nn.Conv2d(n_filters, in_channels, 3, padding=1))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out + x

class ResidualStack(nn.Module):
    def __init__(self, in_channels, n_filters=64, n_blocks=5, tail=False):
        super(ResidualStack, self).__init__()
        self.n_blocks = n_blocks
        self.model = []
        for _ in range(n_blocks):
            self.model.append(ResidualBlock(in_channels, n_filters))
        if not tail:
            self.model.append(nn.ReLU())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)

        self.deconv_layers = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualStack(128, n_blocks=3),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1)
        )

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 64, 4, 4)
        reconstruction = F.sigmoid(self.deconv_layers(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 128, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            ResidualStack(128, n_blocks=2, tail=True)
        )

        self.fc_mu = nn.Linear(4*4*128, latent_size)
        self.fc_logsigma = nn.Linear(4*4*128, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        x = self.conv_layers(x)
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

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
