""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, PixelVAE, Controller
import gym
from gym import spaces
import gym.envs.box2d

# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64
N_COLOR_DIM = 4

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def get_env_id(dataset):
    if dataset == 'carracing':
        return 'CarRacing-v0'
    elif dataset == 'pong':
        return 'Pong-v0'
    else:
        raise Exception('Invalid dataset: {}'.format(dataset))

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, model, is_best, folder_name):
    """ Save state in filename. Also save in best_filename if is_best. """
    checkpoint_fname = join(folder_name, 'checkpoint.tar')
    best_fname = join(folder_name, 'best.tar')
    model_cpk_fname = join(folder_name, 'model_checkpoint.pt')
    model_best_fname = join(folder_name, 'model_best.pt')

    torch.save(state, checkpoint_fname)
    torch.save(model, model_cpk_fname)
    if is_best:
        torch.save(state, best_fname)
        torch.save(model, model_best_fname)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class IncreaseSize(object):
    """ Transform to increase size of Pong ball """
    def __init__(self, game, n_expand=1):
        self.is_pong = game == 'pong'
        self.n_expand = n_expand

        if self.is_pong:
            print("Using pong, increasing ball size")

    def __call__(self, sample):
        if not self.is_pong:
            return sample

        sub_sample = sample[:, 34:-16, :] # now the only white is the ball
        is_white = (sub_sample > 0.9).all(dim=0) # find all white pixels
        is_white = torch.nonzero(is_white)
        if is_white.size(0) == 0: # no ball to resize, so return
            return sample

        tl, br = is_white[0], is_white[-1]
        tl = torch.clamp(tl - self.n_expand, min=0)
        br += self.n_expand
        tlr, tlc = tl
        brr, brc = br

        # Re-shift coordinates to un-subsampled image
        tlr += 34
        brr += 34

        sample[:, tlr:brr+1, tlc:brc+1] = 1.0
        return sample


class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit, dataset):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_folder = join(mdir, dataset, 'vae')
        rnn_folder = join(mdir, dataset, 'rnn')
        ctrl_folder = join(mdir, dataset, 'ctrl')

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (join(vae_folder, 'best.tar'), join(rnn_folder, 'best.tar'))]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = torch.load(join(vae_folder, 'model_best.pt'))
        self.vae.to_device_encoder_only(device)

        self.mdrnn = torch.load(join(rnn_folder, 'model_best.pt')).to(device)

        ctrl_file = join(ctrl_folder, 'best.tar')
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller = torch.load(join(ctrl_folder, 'model_best.pt'))
        else:
            self.controller = Controller(LSIZE, RSIZE, ASIZE)
        self.controller = self.controller.to(device)

        self.env = gym.make('CarRacing-v0')
        self.device = device
        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        latent_mu, _ = self.vae.encode(obs)
        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            obs = transform(obs).unsqueeze(0).to(self.device)
            obs *= 255
            obs = torch.floor(obs / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)

            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                return - cumulative
            i += 1

class PreprocessEnv(object):
    """
    Process multiple environments in batches
    """

    def __init__(self, envs, args, device):
        folder = join(args.logdir, args.dataset)
        assert exists(folder), 'No {} model folder'.format(args.dataset)

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

        vae = torch.load(vae_model)
        vae.to_device_encoder_only(device)
        rnn = torch.load(rnn_model).to(device)

        self.vae, self.rnn = vae, rnn
        self.device = device
        self.envs = envs

        self.action_space = envs.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(LSIZE + RSIZE,),
                                            dtype=np.float32)
        self.n_envs = args.num_processes
        self.hiddens = None

    def reset(self):
        self.hiddens = [
            torch.zeros(self.n_envs, RSIZE).to(self.device)
            for _ in range(2)
        ]
        obs = self.envs.reset()
        z = self._process_obs(obs)
        obs = torch.cat((z, self.hiddens[0]), 1)
        return obs

    def step(self, action):
        assert self.hiddens is not None, 'Must call reset() before step()'
        obs, reward, done, info = self.envs.step(action)
        z = self._process_obs(obs)
        with torch.no_grad():
            action = action.to(self.device)
            _, _, _, _, _, self.hiddens = self.rnn(action, z, self.hiddens)

        self.hiddens[0][done.astype('uint8'), :] = 0
        self.hiddens[1][done.astype('uint8'), :] = 0

        obs = torch.cat((z, self.hiddens[0]), 1)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def close(self):
        self.envs.close()

    def _process_obs(self, obs):
        with torch.no_grad():
            obs = torch.stack([transform(o) for o in obs], 0)
            obs = obs.to(self.device)
            latent_mu, _ = self.vae.encode(obs)
            return latent_mu
