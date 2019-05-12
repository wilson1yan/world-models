"""
Simulated carracing environment.
"""
import argparse
from os.path import join, exists
import torch
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical
import gym
from gym import spaces
from utils.misc import LSIZE, RSIZE, RED_SIZE, IncreaseSize, N_COLOR_DIM
from data.loaders import RolloutObservationDataset

import numpy as np

class SimulatedBoxing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v0 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, directory):
        vae_folder = join(directory, 'vqvae')
        rnn_folder = join(directory, 'cat_rnn_st')

        vae_file = join(vae_folder, 'best.tar')
        rnn_file = join(rnn_folder, 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No RNN model in the directory..."

        # spaces
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3),
                                            dtype=np.uint8)

        self.device = torch.device('cuda')

        # load VAE
        vae_state = torch.load(vae_file, map_location=self.device)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        self._vae = torch.load(join(vae_folder, 'model_best.pt'), map_location=self.device)

        # load MDRNN
        rnn_state = torch.load(rnn_file, map_location=self.device)
        print("Loading CatRNN at epoch {}, "
              "with test error {}...".format(
                  rnn_state['epoch'], rnn_state['precision']))
        self._rnn = torch.load(join(rnn_folder, 'model_best.pt'), map_location=self.device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            IncreaseSize(game=args.dataset, n_expand=2),
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor(),
        ])
        dataset = RolloutObservationDataset(join('datasets', args.dataset),
                                            transform, train=False,
                                            buffer_size=10)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True
        )
        self._loader = loader

        # init state
        self._lstate = torch.randn(1, LSIZE)
        self._hstate = 2 * [torch.zeros(1, RSIZE)]

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
        self._lstate = self._vae.encode(obs)[0]


        # also reset monitor
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))

    def step(self, a):
        """ One step forward """
        with torch.no_grad():
            action = torch.zeros(1, 18).to(self.device)
            action[0, a] = 1
            lstate_embed = self._vae.to_embedding(self._lstate).contiguous()
            lstate_embed = lstate_embed.view(lstate_embed.size(0), -1)
            dist_outs, r, d, n_h = self._rnn(action, lstate_embed, self._hstate)
            self._lstate = self._vae.codebook(dist_outs)
            self._hstate = n_h

            self._obs = self._vae.sample(self._lstate, self.device)
            np_obs = self._obs.cpu().numpy()
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            return np_obs, r.item(), d.item() > 0

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
    parser.add_argument('--dataset', type=str, default='boxing')
    args = parser.parse_args()
    env = SimulatedBoxing(join(args.logdir, args.dataset))

    env.reset()
    action = np.array([0])

    keys_pressed = []

    def get_action():
        if 'down' in keys_pressed and 'left' in keys_pressed and ' ' in keys_pressed:
            return 17
        elif 'down' in keys_pressed and 'right' in keys_pressed and ' ' in keys_pressed:
            return 16
        elif 'up' in keys_pressed and 'left' in keys_pressed and ' ' in keys_pressed:
            return 15
        elif 'up' in keys_pressed and 'right' in keys_pressed and ' ' in keys_pressed:
            return 14
        elif 'down' in keys_pressed and ' ' in keys_pressed:
            return 13
        elif 'left' in keys_pressed and ' ' in keys_pressed:
            return 12
        elif 'right' in keys_pressed and ' ' in keys_pressed:
            return 11
        elif 'up' in keys_pressed and ' ' in keys_pressed:
            return 10
        elif 'down' in keys_pressed and 'left' in keys_pressed:
            return 9
        elif 'down' in keys_pressed and 'right' in keys_pressed:
            return 8
        elif 'up' in keys_pressed and 'left' in keys_pressed:
            return 7
        elif 'up' in keys_pressed and 'right' in keys_pressed:
            return 6
        elif 'down' in keys_pressed:
            return 5
        elif 'left' in keys_pressed:
            return 4
        elif 'right' in keys_pressed:
            return 3
        elif 'up' in keys_pressed:
            return 2
        elif ' ' in keys_pressed:
            return 1
        else:
            return 0

    def on_key_press(event):
        """ Defines key pressed behavior """
        print(event.key)
        keys_pressed.append(event.key)


    def on_key_release(event):
        """ Defines key pressed behavior """
        keys_pressed.remove(event.key)

    env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        action = get_action()
        _, _, done = env.step(action)
        env.render()
        if done:
            break
