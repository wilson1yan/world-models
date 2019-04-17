import argparse
from os import mkdir
from os.path import join, exists

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from utils.misc import LSIZE, RED_SIZE
from data.loaders import RolloutObservationDataset
from models.vae import VAE, PixelVAE

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--model', type=str, default='vae')
parser.add_argument('--beta', type=int, default=1,
                   help='beta for beta-VAE')
parser.add_argument('--n', type=int, default=4,
                    help='n images')

args = parser.parse_args()

N_COLOR_DIM = 4

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
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
    dataset_test, batch_size=args.n, shuffle=True, num_workers=2)

if args.model == 'vae':
    model = VAE(3, LSIZE).to(device)
elif args.model == 'pixel_vae_c':
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=False).to(device)
elif args.model == 'pixel_vae_l':
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=True).to(device)
else:
    raise Exception('Invalid model {}'.format(args.model))

vae_dir = join(args.logdir, '{}_beta{}'.format(args.model, args.beta))
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

with torch.no_grad():
    recon_x1 = model(data)[0]
    recon_x1 = F.softmax(recon_x1, 1)
    recon_x1 = torch.max(recon_x1, 1)[1].float() / (N_COLOR_DIM - 1)
    recon_x1 = recon_x1.cpu()

    z = model.encoder(data)[0]
    recon_x2 = model.sample(z, device)
    recon_x2 = recon_x2.cpu()

data = data.cpu()
images = torch.cat((data, recon_x1, recon_x2), dim=0)
save_image(images, join(vae_dir, 'reconsruction.png'), nrow=args.n)
