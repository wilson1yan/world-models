""" Training VAE """
import argparse
from os.path import join, exists
from os import makedirs

import numpy as np

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.pixel_cnn.models import Gated

from utils.misc import save_checkpoint, IncreaseSize
from utils.misc import LSIZE, RED_SIZE, N_COLOR_DIM
from utils.metrics import compute_mmd

from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged',
                    default='logs')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--dataset', type=str, default='carracing')


args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

latent_size = (4, 4)

transform = transforms.Compose([
    transforms.ToTensor(),
    IncreaseSize(game=args.dataset, n_expand=2),
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_folder = join('datasets', args.dataset)
dataset_train = RolloutObservationDataset(dataset_folder,
                                          transform, train=True,
                                          buffer_size=100)
dataset_test = RolloutObservationDataset(dataset_folder,
                                         transform, train=False,
                                         buffer_size=100)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, args.dataset, 'vqvae')
reload_file = join(vae_dir, 'best.tar')
assert exists(reload_file)
state = torch.load(reload_file, map_location=device)
print("Loading VQVAE at epoch {}"
      ", with test error {}".format(
          state['epoch'],
          state['precision']))
vae = torch.load(join(vae_dir, 'model_best.pt'), map_location=device)
vae = vae.to(device)

prior_dir = join(args.logdir, args.dataset, 'vqprior')
if not exists(prior_dir):
    makedirs(prior_dir)

reload_file = join(prior_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model = torch.load(join(prior_dir, 'model_best.pt'))
    optimizer = optim.Adam(model.parameters(),
                           lr=np.sqrt(args.batch_size / 32) * 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])
else:
    model = Gated((1,) + latent_size, 64, 1, 128, k=3, padding=1)
    optimizer = optim.Adam(model.parameters(),
                           lr=np.sqrt(args.batch_size / 32) * 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)
model = model.to(device)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(latents, out):
    """ VAE loss function """
    loss = F.cross_entropy(out, latents.detach())
    return loss

def to_latent(x):
    with torch.no_grad():
        latents = vae.encode(x)[0].unsqueeze(1)
    return latents

def process(data):
    data *= 255
    data = torch.floor(data / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)
    return data

def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data = process(data)
        optimizer.zero_grad()
        latents = to_latent(data)
        out = model(latents.float())
        loss = loss_function(latents, out)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data = process(data)
            latents = to_latent(data)
            out = model(latents.float())
            test_loss += loss_function(latents, out).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(prior_dir, 'best.tar')
    filename = join(prior_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, model, is_best, prior_dir)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
