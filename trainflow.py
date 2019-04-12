""" Training Flow """
import argparse
from os.path import join, exists
from os import mkdir
import numpy as np

import torch
import torch.utils.data
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torchvision import transforms
from torchvision.utils import save_image

from models.flow import Glow, RealNVP, MultiscaleGlow

from utils.misc import save_checkpoint
from utils.misc import RED_SIZE, ASIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

parser = argparse.ArgumentParser(description='Flow Trainer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()

BATCH = 32

torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

dataset_train = RolloutObservationDataset('datasets/carracing',
                                          transform_train, train=True)
dataset_test = RolloutObservationDataset('datasets/carracing',
                                         transform_test, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

N = ASIZE * RED_SIZE * RED_SIZE
prior = Normal(0, 1)
model = MultiscaleGlow().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

def loss_function(z, logdet):
    """ Flow loss function """
    z = z.view(z.size(0), -1)
    log_prob = prior.log_prob(z).sum(1)
    ll = log_prob + logdet
    loss = -ll.mean() / N
    return loss


def process(data):
    data *= 255
    data = torch.floor(data / 64)
    data += torch.rand_like(data)
    return data

def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = process(data)
        data = data.to(device)

        # x_rec = model.sample(model(data[[0]])[0])
        # print('Invert error', torch.max(torch.abs(x_rec - data[[0]])))

        optimizer.zero_grad()
        for i in range(0, data.size(0), BATCH):
            z, logdet = model(data[i:i+BATCH])

            loss = loss_function(z, logdet)
            loss.backward()
            train_loss += loss.item() / np.log(2)

        for param in optimizer.param_groups:
            nn.utils.clip_grad_norm_(param['params'], 1, 2)

        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
            data = process(data)
            data = data.to(device)
            for i in range(0, data.size(0), BATCH):
                z, logdet = model(data[i:i+BATCH])
                test_loss += loss_function(z, logdet).item() / np.log(2)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check flow dir exists, if not, create it
# flow_dir = join(args.logdir, 'flow')
flow_dir = join('logs', args.logdir)
if not exists(flow_dir):
    mkdir(flow_dir)
    mkdir(join(flow_dir, 'samples'))

reload_file = join(flow_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(flow_dir, 'best.tar')
    filename = join(flow_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(64, *model.latent_size).to(device)
            sample = model.sample(sample).cpu() / 4
            save_image(sample,
                       join(flow_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
