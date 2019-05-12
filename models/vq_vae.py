import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

import models.pixel_cnn.models as models
from models.base import ResBlock

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super(VQEmbedding, self).__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        indices = indices.view(*z_q_x_bar_.size()[:-1])

        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        return z_q_x, z_q_x_bar, indices

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_size, code_dim, out_channels=None):
        super(Decoder, self).__init__()
        self.img_channels = img_size[0]

        self.deconv1 = nn.ConvTranspose2d(code_dim, 128, 6, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 7, stride=2)
        if out_channels is None:
            self.deconv3 = nn.ConvTranspose2d(64, img_size[0], 8, stride=2)
        else:
            self.deconv3 = nn.ConvTranspose2d(64, out_channels, 8, stride=2)
        # if out_channels is None:
        #     self.deconv4 = nn.ConvTranspose2d(32, img_size[0], 2, stride=2)
        # else:
        #     self.deconv4 = nn.ConvTranspose2d(32, out_channels, 2, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
        # reconstruction = self.deconv4(x)
        # return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_size, code_dim, cond_size=None):
        super(Encoder, self).__init__()
        self.img_channels = img_size[0]
        self.cond_size = cond_size
        self.code_dim = code_dim

        self.conv1 = nn.Conv2d(img_size[0], 128, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, code_dim, 3, stride=2, padding=1)

        if cond_size is not None:
            self.fc1 = nn.Linear(cond_size, 4*4*code_dim)

    def forward(self, x, cond=None): # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        if self.cond_size is not None:
            x = F.relu(self.conv5(x) + self.fc1(cond).view(x.size(0),
                                                           self.code_dim, 4, 4))
        else:
            x = F.relu(self.conv5(x))
        return x

class VectorQuantizedVAELarge(nn.Module):
    def __init__(self, img_size, dim, K=256):
        super(VectorQuantizedVAELarge, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_size[0], dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            # nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
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

        self.apply(weights_init)

    def encode_train(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        return z_e_x, z_q_x_st, z_q_x, indices

    def decode_train(self, obs, z_q_x_st):
        return self.decoder(z_q_x_st)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents, None

    def to_embedding(self, latents):
        return self.codebook.embedding(latents).permute(0, 3, 1, 2)

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def sample(self, z, device):
        return self.decode(z)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

class VectorQuantizedVAE(nn.Module):
    def __init__(self, img_size, code_dim, cond_size=None, K=128):
        super(VectorQuantizedVAE, self).__init__()
        self.encoder = Encoder(img_size, code_dim, cond_size=cond_size)
        self.codebook = VQEmbedding(K, code_dim)
        self.decoder = Decoder(img_size, code_dim)

        self.apply(weights_init)

    def encode(self, x, cond=None):
        z_e_x = self.encoder(x, cond=cond)
        latents = self.codebook(z_e_x)
        return latents, None

    def encode_train(self, x, cond=None):
        z_e_x = self.encoder(x, cond=cond)
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        return z_e_x, z_q_x_st, z_q_x, indices

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def to_embedding(self, latents):
        return self.codebook.embedding(latents).permute(0, 3, 1, 2)

    def decode_train(self, obs, z_q_x_st):
        return self.decoder(z_q_x_st)

    def sample(self, z, device):
        return self.decode(z)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, _ = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

class PixelVectorQuantizedVAE(nn.Module):

    def __init__(self, img_size, code_dim, n_color_dims,
                 cond_size=None, K=128):
        super(PixelVectorQuantizedVAE, self).__init__()
        self.encoder = Encoder(img_size, code_dim, cond_size=cond_size)
        self.codebook = VQEmbedding(K, code_dim)
        self.decoder = Decoder(img_size, code_dim, out_channels=128)
        self.pixel_cnn = models.LGated(img_size, 128,
                                       120, num_layers=2,
                                       n_color_dims=n_color_dims,
                                       k=5, padding=2)
        self.img_size = img_size

        self.apply(weights_init)

    def encode(self, x, cond=None):
        z_e_x = self.encoder(x, cond=cond)
        latents = self.codebook(z_e_x)
        return latents, None

    def encode_train(self, x, cond=None):
        z_e_x = self.encoder(x, cond=cond)
        z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        return z_e_x, z_q_x_st, z_q_x, indices

    def decode(self, latents, device):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        z_q_x = z_q_x.view(z_q_x.size(0), -1)
        z = self.decoder(z_q_x)
        x_tilde = self.pixel_cnn.sample(self.img_size, device,
                                        cond=z)
        return x_tilde

    def to_embedding(self, latents):
        return self.codebook.embedding(latents).permute(0, 3, 1, 2)

    def decode_train(self, obs, z_q_x_st):
        z = self.decoder(z_q_x_st)
        return self.pixel_cnn(obs, z)

    def sample(self, z, device):
        return self.decode(z, device)

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x, _ = self.codebook.straight_through(z_e_x)
        z = self.decoder(z_q_x_st)
        x_tilde = self.pixel_cnn(x, z)
        return x_tilde, z_e_x, z_q_x
