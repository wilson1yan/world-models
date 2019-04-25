""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
parser.add_argument('--beta', type=int, default=1,
                   help='beta for beta-VAE')
parser.add_argument('--model', type=str, default='vae')
parser.add_argument('--dataset', type=str, default='carracing')
parser.add_argument('--reg', type=str, default='mmd')
args = parser.parse_args()

N_COLOR_DIM = 4

ctrl_file = join(args.logdir, 'ctrl_{}_{}_beta{}_{}'.format(args.reg, args.model,
                                                           args.beta, args.dataset),
                 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000,
                             args.beta, args.model, args.dataset,
                             args.reg, N_COLOR_DIM)

with torch.no_grad():
    generator.rollout(None, render=True)
