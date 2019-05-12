import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical

import numpy as np

from models.base import MLP, SimpleConv
from models.discrete_vae import DiscreteVAE
from utils.misc import RED_SIZE, N_COLOR_DIM, RSIZE

class SeqVAECat(nn.Module):

    def __init__(self, asize, n_categories):
        super(SeqVAECat, self).__init__()
        self.latent_shape = (8, 8)
        self.n_categories = n_categories
        self.asize = asize
        lsize = np.prod(self.latent_shape) * 128

        self.vae = DiscreteVAE((3, RED_SIZE, RED_SIZE), 128, n_categories,
                               cond_size=RSIZE)
        self.rnn = nn.LSTMCell(lsize + asize, RSIZE)
        self.transition = MLP(RSIZE, np.prod(self.latent_shape) * n_categories,
                              [256, 256])
        self.reward = MLP(RSIZE + lsize, 1, [128])
        self.terminal = MLP(RSIZE + lsize, 1, [128],
                            output_activation=torch.sigmoid)

    def anneal(self):
        self.vae.anneal()

    def encode(self, obs, cond):
        return self.vae.encode(obs, cond=cond)

    def decode(self, latent, device):
        return self.vae.sample(latent, device)

    def step(self, latent, action, hiddens, device):
        latent = self.vae.to_embedding(latent, device)
        flat_latent = latent.view(latent.size(0), -1)
        rnn_input = torch.cat((flat_latent, action), -1)
        hiddens = self.rnn(rnn_input, hiddens)
        next_latent = self.transition(hiddens[0])
        next_latent = next_latent.view(next_latent.size(0),
                                       self.n_categories, *self.latent_shape)
        next_latent = next_latent.permute(0, 2, 3, 1).contiguous()
        next_latent = F.softmax(next_latent / 0.2, -1)
        next_latent = Categorical(next_latent).sample()
        next_latent = next_latent.view(latent.size(0), *self.latent_shape)

        aux_input = torch.cat((flat_latent, hiddens[0]), -1)
        reward = self.reward(aux_input)
        done = self.terminal(aux_input)

        return next_latent, reward, done, hiddens

    def forward(self, obs, actions, hiddens):
        """
        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, RED_SIZE, RED_SIZE)
        :args actions: 3D torch tensor(BSIZE, SEQ_LEN, ASIZE)
        """
        batch_size, seq_len = obs.size(0), obs.size(1)
        obs = torch.transpose(obs, 0, 1)
        actions = torch.transpose(actions, 0, 1)

        obs_latents, trans_latents = [], []
        obs_log_probs = []
        recon_obs = []
        rewards, dones = [], []
        for t in range(seq_len):
            z, log_probs_z = self.vae.encode_train(obs[t], cond=hiddens[0])
            obs_latent = z
            obs_latent_flat = obs_latent.view(obs_latent.size(0), -1)
            out = self.transition(hiddens[0])
            trans_latent = out.view(out.size(0), self.n_categories,
                                    *self.latent_shape)
            trans_latent = F.log_softmax(trans_latent / self.vae.temperature, 1)
            recon = self.vae.decode_train(obs_latent)

            aux_input = torch.cat((obs_latent_flat, hiddens[0]), -1)
            reward = self.reward(aux_input)
            done = self.terminal(aux_input)

            rnn_input = torch.cat((obs_latent_flat, actions[t]), -1)
            hiddens = self.rnn(rnn_input, hiddens)

            obs_latents.append(obs_latent)
            obs_log_probs.append(log_probs_z)
            trans_latents.append(trans_latent)
            recon_obs.append(recon)
            rewards.append(reward)
            dones.append(done)

        obs_latents = torch.stack(obs_latents, 1)
        obs_log_probs = torch.stack(obs_log_probs, 1)
        trans_latents = torch.stack(trans_latents, 1) # Keep class dim 1
        recon_obs = torch.stack(recon_obs, 1)
        rewards = torch.stack(rewards, 1).squeeze(-1)
        dones = torch.stack(dones, 1).squeeze(-1)

        return recon_obs, obs_latents, obs_log_probs, \
               trans_latents, \
               rewards, dones
