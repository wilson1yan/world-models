import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import SimpleConv

def cat_loss(latent_next_obs, dist_outs):
    loss = F.cross_entropy(dist_outs, latent_next_obs, reduce=False)
    loss = loss.view(loss.size(0), -1)
    loss = loss.sum(-1).mean(0)
    return loss

class RNNCat(nn.Module):

    def __init__(self, latent_shape, code_dim, K, actions, hiddens, mode='cm'):
        super(RNNCat, self).__init__()
        lsize = int(np.prod(latent_shape) * code_dim)
        self.rnn = nn.LSTM(lsize + actions, hiddens)
        if mode == 'st':
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) * code_dim + 2)
        elif mode == 'cm':
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) * K + 2)
        else:
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) + 2)

        self.code_dim = code_dim
        self.latent_shape = latent_shape
        self.K = K
        self.n_latents = np.prod(latent_shape)
        self.mode = mode

    def forward(self, actions, latents):
        seq_len, bs = actions.size(0), actions.size(1)
        latents = latents.contiguous()
        latents_flat = latents.view(seq_len, bs, -1)

        outs, _ = self.rnn(torch.cat([latents_flat, actions], 2))
        outs = torch.cat([outs, actions], 2)
        outs = self.output_layer(outs)
        dist_outs = outs[:, :, :-2]

        if self.mode == 'st' or self.mode == 'cm':
            last_dim = self.code_dim if self.mode == 'st' else self.K
            dist_outs = dist_outs.view(dist_outs.size(0), dist_outs.size(1),
                                       *self.latent_shape)
            dist_outs = dist_outs.permute(1, 4, 0, 2, 3) # B x K x SEQ x N_LATENTS
        else:
            dist_outs = dist_outs.view(dist_outs.size(0), dist_outs.size(1),
                                       *self.latent_shape)
            dist_outs = torch.transpose(dist_outs, 0, 1)

        rs = outs[:, :, -2]
        ds = outs[:, :, -1]

        return dist_outs, rs, ds


class RNNCatCell(nn.Module):

    def __init__(self, latent_shape, code_dim, K, actions, hiddens, mode='cm'):
        super(RNNCatCell, self).__init__()
        self.latent_shape = latent_shape
        self.K = K
        self.n_latents = np.prod(latent_shape)
        self.code_dim = code_dim
        self.mode = mode

        lsize = self.n_latents * code_dim

        self.rnn = nn.LSTMCell(lsize + actions, hiddens)
        if mode == 'st':
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) * code_dim + 2)
        elif mode == 'cm':
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) * K + 2)
        else:
            self.output_layer = nn.Linear(hiddens + actions, np.prod(latent_shape) + 2)


    def forward(self, action, latent, hidden):
        next_hidden = self.rnn(torch.cat([latent, action], 1), hidden)
        out_rnn = next_hidden[0]
        out_rnn = torch.cat([out_rnn, action], dim=1)

        outs = self.output_layer(out_rnn)
        dist_out = outs[:, :-2]
        if self.mode == 'st' or self.mode == 'cm':
            last_dim = self.code_dim if self.mode == 'st' else self.K
            dist_out = dist_out.view(dist_out.size(0), *self.latent_shape, self.K)
            dist_out.permute(0, 3, 1, 2)
        else:
            dist_out = dist_out.view(dist_out.size(0), *self.latent_shape)
        r = outs[:, -2]
        d = outs[:, -1]

        return dist_out, r, d, next_hidden
