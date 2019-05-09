import torch.nn as nn
import torch

import numpy as np

from models.base import MLP, SimpleConv
from models.vq_vae import VectorQuantizedVAE
from utils.misc import RED_SIZE, N_COLOR_DIM, RSIZE

class SeqVAECat(nn.Module):

    def __init__(self, asize, vae_model_name, code_dim, K):
        super(SeqVAECat, self).__init__()
        self.latent_shape = (4, 4)
        self.code_dim = code_dim
        self.K = K
        lsize = np.prod(self.latent_shape) * code_dim

        if vae_model_name == 'vae':
            self.vae = VectorQuantizedVAE((3, RED_SIZE, RED_SIZE), code_dim,
                                          cond_size=RSIZE, K=K)
        else:
            raise Exception()
        self.rnn = nn.LSTMCell(lsize + asize, RSIZE)
        self.transition = MLP(RSIZE, np.prod(self.latent_shape) * K, [128])
        self.reward = MLP(RSIZE + lsize, 1, [128])
        self.terminal = MLP(RSIZE + lsize, 1, [128],
                            output_activation=torch.sigmoid)

    def encode(self, obs, cond):
        return self.vae.encode(obs, cond=cond)

    def decode(self, latent, device):
        return self.vae.sample(latent, device)

    def step(self, latent, action, hiddens):
        flat_latent = self.vae.to_embedding(latent)
        flat_latent = flat_latent.view(flat_latent.size(0), -1)
        rnn_input = torch.cat((flat_latent, action), -1)
        hiddens = self.rnn(rnn_input, hiddens)
        next_latent = self.transition(hiddens[0])
        next_latent = next_latent.view(next_latent.size(0),
                                       self.K, *self.latent_shape)
        _, next_latent = torch.max(next_latent, 1)

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
        obs_embeds, idx_targets = [], []
        recon_obs = []
        rewards, dones = [], []
        for t in range(seq_len):
            z_e_x, z_q_x_st, z_q_x, indices = self.vae.encode_train(obs[t],
                                                                    cond=hiddens[0])
            obs_latent = z_q_x_st
            obs_latent_flat = obs_latent.view(obs_latent.size(0), -1)
            out = self.transition(hiddens[0])
            trans_latent = out.view(out.size(0), self.K, *self.latent_shape)
            recon = self.vae.decode_train(obs[t], obs_latent)

            aux_input = torch.cat((obs_latent_flat, hiddens[0]), -1)
            reward = self.reward(aux_input)
            done = self.terminal(aux_input)

            rnn_input = torch.cat((obs_latent_flat, actions[t]), -1)
            hiddens = self.rnn(rnn_input, hiddens)

            obs_latents.append(z_e_x)
            obs_embeds.append(z_q_x)
            trans_latents.append(trans_latent)
            recon_obs.append(recon)
            rewards.append(reward)
            dones.append(done)
            idx_targets.append(indices)

        obs_latents = torch.stack(obs_latents, 1)
        obs_embeds = torch.stack(obs_embeds, 1)
        trans_latents = torch.stack(trans_latents, 2) # Keep class dim 1
        idx_targets = torch.stack(idx_targets, 1)
        recon_obs = torch.stack(recon_obs, 1)
        rewards = torch.stack(rewards, 1).squeeze(-1)
        dones = torch.stack(dones, 1).squeeze(-1)

        return recon_obs, obs_latents, obs_embeds, \
               trans_latents, idx_targets, \
               rewards, dones
