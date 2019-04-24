import argparse
from os import mkdir
from os.path import join, exists

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from utils.misc import LSIZE, RED_SIZE, IncreaseSize
from data.loaders import RolloutObservationDataset
from models.pixel_cnn.models import Gated
from models.vae import VAE, PixelVAE

parser = argparse.ArgumentParser(description='Visualize dataset')
parser.add_argument('--dataset', type=str, default='carracing')

args = parser.parse_args()

N_COLOR_DIM = 4

transform_test = transforms.Compose([
    transforms.ToTensor(),
    IncreaseSize(game=args.dataset, n_expand=2),
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])
dataset_test = RolloutObservationDataset(join('datasets', args.dataset),
                                         transform_test, train=False, buffer_size=1)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=64, shuffle=True, num_workers=2)

data = next(iter(test_loader))
data = torch.floor(data * 255 / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)

save_image(data, 'dataset.png', nrow=8)
