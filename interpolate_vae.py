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

from models.vae import VAE, PixelVAE, AFPixelVAE

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
parser.add_argument('--dataset', type=str, default='carracing',
                    help='dataset name')
parser.add_argument('--reg', type=str, default='kl',
                    help='regularizing distance function')


args = parser.parse_args()

N_COLOR_DIM = 4
N_INTERPOLATE = 4

beta = args.beta
cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_test = RolloutObservationDataset('datasets/carracing',
                                         transform_test, train=False)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=True, num_workers=2)

if args.model == 'vae':
    model = VAE(3, LSIZE).to(device)
elif args.model == 'pixel_vae_c':
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=False).to(device)
elif args.model == 'pixel_vae_l':
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=True).to(device)
# elif args.model == 'pixel_vae_af_c':
#     model = AFPixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
#                        N_COLOR_DIM, upsample=False).to(device)
# elif args.model == 'pixel_vae_af_l':
#     model = AFPixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
#                        N_COLOR_DIM, upsample=True).to(device)
else:
    raise Exception('Invalid model {}'.format(args.model))

vae_dir = join(args.logdir, '{}_{}_beta{}_{}'.format(args.reg, args.model,
                                                     args.beta,
                                                     args.dataset))
assert exists(vae_dir)

reload_file = join(vae_dir, 'best.tar')
state = torch.load(reload_file)
print("Reloading model at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
model.load_state_dict(state['state_dict'])

data = next(iter(test_loader))
data = data.to(device)
data = torch.floor(data * 255 / 64) / (N_COLOR_DIM - 1)

z = model.encoder(data)[0]
print(z)
idx = 0

interp_vals = np.linspace(-1, 1, N_INTERPOLATE)
zs = [z for _ in range(N_INTERPOLATE)]
for i in range(N_INTERPOLATE):
    zs[i][:, idx] = interp_vals[i]
zs = torch.stack(zs, dim=1).view(4 * N_INTERPOLATE, z.size(1))
recon_x = model.sample(zs, device).cpu()

save_image(recon_x, join(vae_dir, 'latent_interpolation.png'), nrow=4)
