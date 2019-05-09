import argparse
from os.path import exists, join
from tqdm import tqdm

import gym

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.distributions.categorical import Categorical

from utils.misc import get_env_id, IncreaseSize
from utils.misc import RED_SIZE, N_COLOR_DIM
from utils.misc import LSIZE, ASIZE, RSIZE
from models.mdrnn import MDRNN
from data.loaders import RolloutObservationDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where everything is stored.',
                        default='logs')
    parser.add_argument('--dataset', type=str, default='carracing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folder = join(args.logdir, args.dataset)
    vae_folder = join(folder, 'seq_vae')
    assert exists(join(vae_folder, 'best.tar'))

    vae = torch.load(join(vae_folder, 'model_best.pt'))
    vae = vae.to(device)

    hiddens = [
        torch.zeros(1, RSIZE).to(device)
        for _ in range(2)
    ]

    n_timesteps = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        IncreaseSize(game=args.dataset, n_expand=2),
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = RolloutObservationDataset(join('datasets', args.dataset),
                                        transform, train=False, buffer_size=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1)

    obs = next(iter(data_loader)).to(device)
    obs = vae.encode(obs, cond=hiddens[0])[0]

    observations = []
    for t in tqdm(range(n_timesteps)):
        observations.append(obs.clone().cpu())
        action = torch.zeros(6).unsqueeze(0).to(device)
        action[0, 3] = 1
        with torch.no_grad():
            obs, r, d, hiddens = vae.step(obs, action, hiddens)
    observations = torch.cat(observations, 0)

    batch_size = 128
    images = []
    for i in range(0, len(observations), batch_size):
        with torch.no_grad():
            obs = observations[i:i+batch_size].to(device)
            images.append(vae.decode(obs, device).cpu())
    images = torch.cat(images, 0)

    save_image(images, join(folder, 'dynamics_images.png'), nrow=8)

    # to_pil = transforms.ToPILImage()
    # images = [to_pil(o) for o in images]
    # images[0].save(join(folder, 'dynamics.gif'), format='GIF',
    #                append_images=images[1:],
    #                duration=100, loop=0)

if __name__ == '__main__':
    main()
