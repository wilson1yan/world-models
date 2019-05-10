""" Training VAE """
import argparse
from os.path import join, exists
from os import makedirs

import numpy as np

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.pixel_cnn.models import Gated

from utils.misc import save_checkpoint, IncreaseSize
from utils.misc import LSIZE, RED_SIZE, N_COLOR_DIM
from utils.metrics import compute_mmd

from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--logdir', type=str, help='Directory where results are logged',
                    default='logs')
parser.add_argument('--dataset', type=str, default='carracing')
parser.add_argument('--n', type=int, default=32)


args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

latent_size = (4, 4)

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, args.dataset, 'vqvae')
reload_file = join(vae_dir, 'best.tar')
assert exists(reload_file)
state = torch.load(reload_file, map_location=device)
print("Loading VQVAE at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
vae = torch.load(join(vae_dir, 'model_best.pt'), map_location=device)
vae = vae.to(device)

prior_dir = join(args.logdir, args.dataset, 'vqprior')
prior_file = join(prior_dir, 'best.tar')
assert exists(prior_file)
state = torch.load(prior_file, map_location=device)
print("Loading Prior at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']
      ))
prior = torch.load(join(prior_dir, 'model_best.pt'), map_location=device)
latents = prior.sample(args.n, (1,) + latent_size, device).long().squeeze(1)
samples = vae.sample(latents, device)
print(samples.size())

save_image(samples, join(vae_dir, 'samples.png'), nrow=8)
