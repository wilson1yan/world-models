import argparse
from os import mkdir
from os.path import join, exists

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from utils.misc import LSIZE, RED_SIZE
from data.loaders import RolloutObservationDataset
from models.pixel_cnn.models import Gated
from models.vae import VAE, PixelVAE

parser = argparse.ArgumentParser(description='Visualize gradient w.r.t input')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--model', type=str, default='pixel_cnn')

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
    dataset_test, batch_size=1, shuffle=True, num_workers=2)

data = next(iter(test_loader))
data = data.to(device)
data.requires_grad = True

if args.model == 'pixel_cnn':
    model = Gated((3, RED_SIZE, RED_SIZE), 120, n_color_dims=N_COLOR_DIM,
                  num_layers=7, k=7, padding=7//2).to(device)
    out = model(data)
elif args.model.startswith('pixel_vae_c'):
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=False).to(device)
    out = model(data)[0]
elif args.model.startswith('pixel_vae_l'):
    model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                     N_COLOR_DIM, upsample=True).to(device)
    out = model(data)[0]
else:
    raise Exception()

out[0, data[0, 0, 14, 14].long(), 0, 14, 14].backward()
grad = data.grad.detach().cpu()[0]
grad = torch.max(torch.abs(grad), dim=0)[0]
grad /= torch.max(grad)

print(grad[13, 13], grad[13, 14])
print(grad[14, 13], grad[14, 14])

model_dir = join(args.logdir, args.model)
save_image(grad, join(model_dir, 'gradient.png'))
