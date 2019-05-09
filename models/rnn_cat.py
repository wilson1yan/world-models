import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def cat_loss(latent_next_obs, dist_outs):
    loss = F.cross_entropy(dist_outs, latent_next_obs, reduce=False)
    loss = loss.view(loss.size(0), -1)
    loss = loss.sum(-1).mean(0)
    return loss

class RNNCat(nn.Module):

    def __init__(self, latent_shape, code_dim, K, actions, hiddens):
        super(RNNCat, self).__init__()
        lsize = int(np.prod(latent_shape) * code_dim)
        self.rnn = nn.LSTM(lsize + actions, hiddens)
        self.output_layer = nn.Linear(hiddens, np.prod(latent_shape) * K + 2)

        self.code_dim = code_dim
        self.latent_shape = latent_shape
        self.K = K
        self.n_latents = np.prod(latent_shape)

    def forward(self, actions, latents):
        seq_len, bs = actions.size(0), actions.size(1)
        latents = latents.contiguous()
        latents_flat = latents.view(latents.size(0), latents.size(1), -1)

        ins = torch.cat([actions, latents_flat], 2)
        outs, _ = self.rnn(ins)
        outs = self.output_layer(outs)
        dist_outs = outs[:, :, :-2]
        dist_outs = dist_outs.view(dist_outs.size(0), dist_outs.size(1), *self.latent_shape, self.K)
        dist_outs = dist_outs.permute(1, 4, 0, 2, 3) # B x K x SEQ x N_LATENTS

        rs = outs[:, :, -2]
        ds = outs[:, :, -1]

        return dist_outs, rs, ds


class RNNCatCell(nn.Module):

    def __init__(self, latent_shape, code_dim, K, actions, hiddens):
        super(RNNCatCell, self).__init__()
        self.latent_shape = latent_shape
        self.K = K
        self.n_latents = np.prod(latent_shape)

        lsize = self.n_latents * code_dim

        self.rnn = nn.LSTMCell(lsize + actions, hiddens)
        self.output_layer = nn.Linear(hiddens, self.n_latents * K + 2)


    def forward(self, action, latent, hidden):
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        outs = self.output_layer(out_rnn)
        dist_out = outs[:, :-2]
        dist_out = dist_out.view(dist_out.size(0), *self.latent_shape, self.K)
        r = outs[:, -2]
        d = outs[:, -1]

        return dist_out.permute(0, 3, 1, 2), r, d, next_hidden
