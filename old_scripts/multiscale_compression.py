import argparse
from os.path import join, exists

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.flow import Glow, RealNVP, MultiscaleGlow
from utils.misc import RED_SIZE, ASIZE
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='Multiscale Compression')
parser.add_argument('--levels', type=int, default=4)
parser.add_argument('--logdir', type=str, default='multiscale_glow')

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if cuda else 'cpu')

flow_dir = join('logs', args.logdir)
assert exists(flow_dir), 'Directory does not exist: {}'.format(flow_dir)
reload_file = join(flow_dir, 'best.tar')
assert exists(reload_file), 'File does not exist: {}'.format(reload_file)

if args.logdir == 'glow':
    model = Glow()
elif args.logdir == 'real_nvp':
    model = RealNVP()
elif args.logdir == 'multiscale_glow':
    model = MultiscaleGlow()


states = torch.load(reload_file)
model.load_state_dict(states['state_dict'])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset = RolloutObservationDataset('datasets/carracing', transform_test,
                                    train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                          shuffle=True)
def process(data):
    data *= 255
    data = torch.floor(data / 64)
    data += torch.rand_like(data)
    return data

data = next(iter(data_loader))
data = process(data)
z, _ = model(data)

zs = []
for level in range(args.levels):
    z_copy = z.clone()
    old_shape = z_copy.size()[1:]
    z_copy = z_copy.view(z_copy.size(0), -1)
    total = z_copy.size(1)
    keep = 0.5 ** level
    if keep < 1:
        z_copy[:, :-int(total * keep)] = torch.randn(z.size(0), total - int(total * keep))
    z_copy = z_copy.view(z_copy.size(0), *old_shape)
    zs.append(z_copy)
z = torch.cat(zs, dim=0)
x = model.sample(z).cpu() / 4

save_image(x, join(flow_dir, 'compression.png'),
           nrow=10)
