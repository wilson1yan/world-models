import argparse
from os import mkdir
from os.path import join, exists

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vq_vae import VectorQuantizedVAE
from utils.misc import LSIZE, RED_SIZE, IncreaseSize, N_COLOR_DIM
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--logdir', type=str, help='Directory where results are logged',
                    default='logs')
parser.add_argument('--n', type=int, default=16,
                    help='n images')
parser.add_argument('--dataset', type=str, default='carracing',
                    help='dataset name')
parser.add_argument('--model', type=str, default='vae')
args = parser.parse_args()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

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
    dataset_test, batch_size=args.n, shuffle=True, num_workers=2)

vae_dir = join(args.logdir, args.dataset, args.model)
assert exists(vae_dir)

reload_file = join(vae_dir, 'best.tar')
state = torch.load(reload_file)
print("Reloading model at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
model = torch.load(join(vae_dir, 'model_best.pt')).to(device)

data = next(iter(test_loader))
data = data.to(device)
data = torch.floor(data * 255 / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)

with torch.no_grad():
    z = model.encode(data)[0]
    recon_x2 = model.sample(z, device)
    recon_x2 = recon_x2.cpu()

    # z = torch.randn(args.n, LSIZE).to(device)
    # samples = model.sample(z, device)
    # samples = samples.cpu()

data = data.cpu()
images = torch.cat((data, recon_x2), dim=0)
save_image(images, join(vae_dir, 'reconstruction.png'), nrow=args.n)
