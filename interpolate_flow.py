import argparse

import torch
import torch.utils.data
from torchvision.utils import save_image
import torchvision.transforms as transforms

from utils.misc import RED_SIZE, ASIZE
from models.flow import Glow, RealNVP, MultiscaleGlow
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='Interpolate Flow')
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--logdir', type=str)
parser.add_argument('--interpolates', type=int, default=5)

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if cuda else 'cpu')

if args.logdir == 'glow':
    model = Glow()
elif args.logdir == 'real_nvp':
    model = RealNVP()
elif args.logdir == 'multiscale_glow':
    model = MultiscaleGlow()

flow_dir = join('logs', args.logdir)
assert exists(flow_dir), 'Directory does not exist'.format(flow_dir)
reload_file = join(flow_dir, 'best.tar')
assert exists(reload_file, 'File does not exist'.format(reload_file))

states = torch.load(reload_file)
model.load_state_dict(state['state_dict'])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset = RolloutObservationDataset('datasets/carracing', transform_test,
                                    train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n * 2,
                                          shuffle=True)

def process(data):
    data *= 255
    data = torch.floor(data / 64)
    data += torch.rand_like(data)
    return data

data = next(iter(data_loader))
data = process(data)
z, _ = model(data)

xs = []
for i in range(args.n):
    z1, z2 = data[2*i], data[2*i+1]
    zs = [(z2 - z1) * j / (args.interpolates - 1)
          for j in range(args.interpolates)]
    zs = torch.stack(zs, dim=0)

    x = model.sample(zs).cpu() / 4
    xs.apend(x)

xs = torch.cat(xs, dim=0)
save_image(xs, join(flow_dir, 'interpolation.png'),
           nrow=args.interpolates)
