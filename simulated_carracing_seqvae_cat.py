"""
Simulated carracing environment.
"""
import argparse
from os.path import join, exists

import torch
from torch.distributions.categorical import Categorical
import torchvision.transforms as transforms
from torchvision.utils import save_image

import gym
from gym import spaces

from models.seq_vae import SeqVAE
from models.mdrnn import MDRNNCell
from utils.misc import LSIZE, RSIZE, RED_SIZE, N_COLOR_DIM, IncreaseSize
from data.loaders import RolloutObservationDataset

import numpy as np

class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, directory):
        vae_folder = join(directory, 'seq_vae_cat')
        vae_file = join(vae_folder, 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."

        # spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3),
                                            dtype=np.uint8)

        self.device = torch.device('cuda')

        # load VAE
        vae_state = torch.load(vae_file)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        self._vae = torch.load(join(vae_folder, 'model_best.pt')).to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            IncreaseSize(game='boxing', n_expand=2),
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor(),
        ])

        self._dataset = RolloutObservationDataset(join('datasets', 'carracing'),
                                                  transform, train=True,
                                                  buffer_size=10)
        self._loader = torch.utils.data.DataLoader(
            self._dataset, batch_size=1, shuffle=True
        )

        # init state
        self._lstate = torch.randn(1, LSIZE).to(self.device)
        self._hstate = 2 * [torch.zeros(1, RSIZE).to(self.device)]

        # obs
        self._obs = None
        self._visual_obs = None

        # rendering
        self.monitor = None
        self.figure = None


    def reset(self):
        """ Resetting """
        import matplotlib.pyplot as plt
        self._hstate = 2 * [torch.zeros(1, RSIZE).to(self.device)]
        obs = next(iter(self._loader)).to(self.device)
        obs *= 255
        obs = torch.floor(obs / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)
        self._lstate = self._vae.encode(obs, self._hstate[0])[0]

        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))

    def step(self, action):
        """ One step forward """
        with torch.no_grad():
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            self._lstate, r, d, self._hstate = self._vae.step(self._lstate, action,
                                                              self._hstate, self.device)
            self._obs = self._vae.decode(self._lstate, self.device)
            np_obs = self._obs.cpu().numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, r.item(), False

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.', default='logs')
    parser.add_argument('--dataset', type=str, default='carracing')
    args = parser.parse_args()
    env = SimulatedCarracing(join(args.logdir, 'carracing'))

    env.reset()
    action = np.array([0., 0., 0.])

    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 1
        if event.key == 'down':
            action[2] = .8
        if event.key == 'left':
            action[0] = -1
        if event.key == 'right':
            action[0] = 1

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[2] = 0
        if event.key == 'left' and action[0] == -1:
            action[0] = 0
        if event.key == 'right' and action[0] == 1:
            action[0] = 0

    env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        _, _, done = env.step(action)
        env.render()
        if done:
            break
