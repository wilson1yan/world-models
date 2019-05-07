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

from models.vae import VAE, PixelVAE, AFPixelVAE

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
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')
parser.add_argument('--model', type=str, default='vae')
parser.add_argument('--dataset', type=str, default='carracing')


args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


transform_train = transforms.Compose([
    transforms.ToTensor(),
    IncreaseSize(game=args.dataset, n_expand=2),
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    IncreaseSize(game=args.dataset, n_expand=2),
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_folder = join('datasets', args.dataset)
dataset_train = RolloutObservationDataset(dataset_folder,
                                          transform_train, train=True,
                                          buffer_size=100)
dataset_test = RolloutObservationDataset(dataset_folder,
                                         transform_test, train=False,
                                         buffer_size=100)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, args.dataset, 'vae')
if not exists(vae_dir):
    makedirs(vae_dir)
    makedirs(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model = torch.load(join(vae_dir, 'model_best.pt'))
    optimizer = optim.Adam(model.parameters(),
                           lr=np.sqrt(args.batch_size / 32) * 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])
else:
    if args.model == 'vae':
        model = VAE((3, RED_SIZE, RED_SIZE), LSIZE)
    elif args.model == 'pixel_vae':
        model = PixelVAE((3, RED_SIZE, RED_SIZE), LSIZE,
                         N_COLOR_DIM, upsample=False)
    else:
        raise Exception('Invalid model {}'.format(args.model))

    optimizer = optim.Adam(model.parameters(),
                           lr=np.sqrt(args.batch_size / 32) * 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    earlystopping = EarlyStopping('min', patience=30)

model = model.to(device)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, out):
    """ VAE loss function """
    recon_x, mu, logsigma, z = out

    # if args.model.startswith('pixel_vae'):
    #     target = (x * (N_COLOR_DIM - 1)).long()
    #     BCE = F.cross_entropy(recon_x, target, reduce=False).view(x.size(0), -1).sum(-1)
    #     BCE = BCE.mean()
    #     BCE /= 3 * RED_SIZE * RED_SIZE
    # else:
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # true_samples = torch.randn(*z.size()).to(device)
    # KLD = compute_mmd(z, true_samples)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

    return BCE + KLD, BCE, KLD

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
        out = model(data)
        loss, bce, kld = loss_function(data, out)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, BCE: {:.3f}, KLD: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), bce.item() / len(data), kld.item() / len(data)))

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
            out = model(data)
            test_loss += loss_function(data, out)[0].item()

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
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, model, is_best, vae_dir)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(16, LSIZE).to(device)
            sample = model.sample(sample, device).cpu()
            save_image(sample,
                       join(vae_dir, 'samples/sample_' + str(epoch - 1) + '.png'),
                       nrow=4)
