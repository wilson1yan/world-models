
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pixel_cnn import PixelCNN

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

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
    def __init__(self, img_channels, latent_size, n_color_dim):
        super(PixelVAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
        self.pixel_cnn = PixelCNN(img_channels, n_color_dim)
        self.n_color_dim = n_color_dim

    def sample(self, z):
        z = self.decoder(z)
        images = torch.zeros_like(z)

        for r in range(images.size(2)):
            for c in range(images.size(3)):
                out = self.pixel_cnn(images, z)
                for channel in range(3):
                    probs = F.softmax(out[:, :, channel, r, c], 1).data
                    pixel_sample = torch.multinomial(probs, 1).float() / (self.n_color_dim - 1)
                    images[:, :, r, c] = pixel_sample
        return images

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        z = self.decoder(z)
        recon_x = self.pixel_cnn(x, z)
        return recon_x, mu, logsigma
