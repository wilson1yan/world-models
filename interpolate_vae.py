import argparse
from os.path import join, exists
from os import mkdir

import numpy as np

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vae import VAE, PixelVAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Interplation Visualizer')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--beta', type=int, default=1,
                   help='beta for beta-VAE')
parser.add_argument('--model', type=str, default='vae')


args = parser.parse_args()
beta = args.beta
cuda = torch.cuda.is_available()
