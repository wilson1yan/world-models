import torch.nn as nn
import torch

from models.mlp import MLP
from models.vae import VAE, PixelVAE
from utils.misc import LSIZE, RED_SIZE, N_COLOR_DIM, RSIZE

class SeqVAE(nn.Module):

    def __init__(self, asize, vae_model_name):
        super(SeqVAE, self).__init__()

        if vae_model_name == 'vae':
            self.vae = VAE((3, RED_SIZE, RED_SIZE), LSIZE, cond_size=RSIZE)
        elif vae_model_name == 'pixel_vae':
            self.vae = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                                N_COLOR_DIM, cond_size=RSIZE)
        else:
            raise Exception()
        self.rnn = nn.LSTMCell(LSIZE + asize, RSIZE)
        self.transition = MLP(RSIZE, LSIZE * 2, [128])
        self.reward = MLP(RSIZE + LSIZE, 1, [128])
        self.terminal = MLP(RSIZE + LSIZE, 1, [128],
                            output_activation=torch.sigmoid)

    def encode(self, obs, cond):
        return self.vae.encode(obs, cond=cond)

    def decode(self, latent, device):
        return self.vae.sample(latent, device)

    def step(self, latent, action, hiddens):
        rnn_input = torch.cat((latent, action), -1)
        hiddens = self.rnn(rnn_input, hiddens)
        next_latent = self.transition(hiddens[0]).chunk(2, 1)[0]

        aux_input = torch.cat((latent, hiddens[0]), -1)
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
        obs_logsigmas, trans_logsigmas = [], []
        recon_obs = []
        rewards, dones = [], []
        for t in range(seq_len):
            obs_latent, obs_logsigma = self.vae.encoder(obs[t], cond=hiddens[0])
            obs_latent = obs_latent
            out = self.transition(hiddens[0])
            trans_latent, trans_logsigma = out.chunk(2, 1)
            trans_logsigma = torch.tanh(trans_logsigma)
            recon = torch.sigmoid(self.vae.decoder(obs_latent))

            aux_input = torch.cat((obs_latent, hiddens[0]), -1)
            reward = self.reward(aux_input)
            done = self.terminal(aux_input)

            rnn_input = torch.cat((obs_latent, actions[t]), -1)
            hiddens = self.rnn(rnn_input, hiddens)

            obs_latents.append(obs_latent)
            obs_logsigmas.append(obs_logsigma)
            trans_latents.append(trans_latent)
            trans_logsigmas.append(trans_logsigma)
            recon_obs.append(recon)
            rewards.append(reward)
            dones.append(done)

        obs_latents = torch.stack(obs_latents, 1)
        obs_logsigmas = torch.stack(obs_logsigmas, 1)
        trans_latents = torch.stack(trans_latents, 1)
        trans_logsigmas = torch.stack(trans_logsigmas, 1)
        recon_obs = torch.stack(recon_obs, 1)
        rewards = torch.stack(rewards, 1).squeeze(-1)
        dones = torch.stack(dones, 1).squeeze(-1)

        return recon_obs, obs_latents, obs_logsigmas, \
               trans_latents, trans_logsigmas, \
               rewards, dones
