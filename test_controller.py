""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.',
                    default='logs')
parser.add_argument('--dataset', type=str, default='carracing')
args = parser.parse_args()

ctrl_file = join(args.logdir, args.dataset, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

generator = RolloutGenerator(args.logdir, device, 1000, args.dataset)

with torch.no_grad():
    generator.rollout(None, render=True)
