import argparse
from os.path import exists, join

import gym

import torch
import torchvision.transforms as transforms

from utils.misc import get_env_id, IncreaseSize
from utils.misc import RED_SIZE, N_COLOR_DIM
from utils.misc import LSIZE, ASIZE, RSIZE
from models.mdrnn import MDRNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where everything is stored.',
                        default='logs')
    parser.add_argument('--dataset', type=str, default='carracing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder = join(mdir, dataset)
    vae_folder = join(folder, 'vae')
    rnn_folder = join(folder, 'rnn')

    assert exists(join(vae_folder, 'best.tar'))
    assert exists(join(rnn_folder, 'best.tar'))

    vae_state = torch.load(join(vae_folder, 'best.tar'))
    rnn_state = torch.load(join(rnn_folder, 'best.tar'))

    for m, s in (('VAE', vae_state), ('RNN', rnn_state)):
        print("Loading {} at epoch {} "
              "with test loss {}".format(
                  m, s['epoch'], s['precision']))

    vae = torch.load(join(vae_folder, 'model_best.pt'))
    rnn = torch.load(join(rnn_folder, 'model_best.pt'))

    vae = vae.to(device)
    rnn = rnn.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        IncreaseSize(game=args.dataset, n_expand=2),
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor(),
    ])

    env_id = get_env_id(args.dataset)
    env = gym.make(env_id)
    obs = env.reset()
    obs = transform(obs).unsqueeze(0).to(device)
    obs = torch.floor(obs / (2 ** 8 / N_COLOR_DIM) / N_COLOR_DIM - 1)
    obs = vae.encode(obs)[0]

    hidden = [
        torch.zeros(1, RSIZE).to(device)
        for _ in range(2)
    ]

    n_timesteps = 1000
    t = 0
    while t < n_timesteps:
        action = env.action_space.sample()
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        obs, _, _, _, _, hidden = rnn(action, obs, hidden)
        
