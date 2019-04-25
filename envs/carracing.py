from os.path import exists, join

import gym
from gym import spaces
import torch
import numpy as np

from utils.misc import LSIZE, RED_SIZE, RSIZE
from utils.misc import transform

class CarRacingEnv(gym.Env):

    def __init__(self, device, time_limit):
        self.env = gym.make('CarRacing-v0')
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(LSIZE + RSIZE,), dtype=np.float32)

        folder = join('logs', 'carracing')
        assert exists(folder), 'No carracing model folder'

        vae_file = join(folder, 'vae', 'best.tar')
        vae_model = join(folder, 'vae', 'model_best.pt')
        rnn_file = join(folder, 'rnn', 'best.tar')
        rnn_model = join(folder, 'rnn', 'model_best.pt')

        vae_state, rnn_state = [
            torch.load(fname)
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = torch.load(vae_model).to(device)
        self.rnn = torch.load(rnn_model).to(device)

        self.time_limit = time_limit
        self.device = device
        self.hidden = None

    def render(self):
        return self.env.render()

    def reset(self):
        self.hidden = torch.zeros(1, RSIZE).to(self.device)
        obs = self.env.reset()
        z = self._process_obs(obs)
        obs = torch.cat((z[0], self.hidden[0])).cpu().numpy()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        z = self._process_obs(obs)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.rnn(action, z, self.hidden)
        obs = torch.cat((z[0], self.hidden[0])).cpu().numpy()
        return obs, reward, done, info

    def _process_obs(self, obs):
        with torch.no_grad():
            obs = transform(obs).unsqueeze(0).to(self.device)
            latent_mu, _ = self.vae.encode(obs)
            return latent_mu
