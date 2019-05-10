""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import makedirs
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from models.rnn_cat import RNNCat, RNNCatCell, cat_loss
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE, IncreaseSize, N_COLOR_DIM
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from torchvision.utils import save_image

parser = argparse.ArgumentParser("RNNCat training")
parser.add_argument('--logdir', type=str, default='logs',
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
parser.add_argument('--dataset', type=str, default='carracing')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants
BSIZE = 16
SEQ_LEN = 32
epochs = 1000

latent_shape = (16, 16)
code_dim = 128
K = 128

# Loading VAE
vae_dir = join(args.logdir, args.dataset, 'vqvae')
vae_file = join(vae_dir, 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = torch.load(join(vae_dir, 'model_best.pt')).to(device)

# Loading model
rnn_dir = join(args.logdir, args.dataset, 'cat_rnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    makedirs(rnn_dir)

mdrnn = RNNCat(latent_shape, code_dim, K, ASIZE, RSIZE).to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    tmp = torch.load(join(rnn_dir, 'model_best.pt'))
    mdrnn.load_state_dict({k+'_l0': v for k, v in tmp.state_dict().items()})
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])



# Data Loading
# transform = transforms.Lambda(
#     lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
transform = transforms.Compose([
    transforms.ToTensor(),
    IncreaseSize(game=args.dataset, n_expand=2),
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

train_loader = DataLoader(
    RolloutSequenceDataset(join('datasets', args.dataset), SEQ_LEN,
                           transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset(join('datasets', args.dataset), SEQ_LEN,
                           transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        batch_size, seq_len = obs.size(0), obs.size(1)
        obs = obs.view(batch_size * seq_len, *obs.size()[2:])
        next_obs = next_obs.view(batch_size * seq_len, *next_obs.size()[2:])

        indices_obs = vae.encode(obs)[0].long()
        indices_next_obs = vae.encode(next_obs)[0].long()

        latent_obs = vae.to_embedding(indices_obs)
        latent_next_obs = vae.to_embedding(indices_next_obs)

        latent_obs = latent_obs.view(batch_size, seq_len, *latent_obs.size()[1:])
        latent_next_obs = latent_next_obs.view(batch_size, seq_len, *latent_next_obs.size()[1:])

        indices_obs = indices_obs.view(batch_size, seq_len, *indices_obs.size()[1:])
        indices_next_obs = indices_next_obs.view(batch_size, seq_len, *indices_next_obs.size()[1:])

    return latent_obs, latent_next_obs, indices_obs, indices_next_obs

def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool,
             indices_obs, indices_next_obs):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action,\
        reward, terminal  = [arr.transpose(1, 0)
                             for arr in [latent_obs, action,
                                       reward, terminal]]
    dist_outs, rs, ds = mdrnn(action, latent_obs)
    gmm = cat_loss(indices_next_obs, dist_outs)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

def process(data):
    data *= 255
    return torch.floor(data / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)

def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]
        obs, next_obs = process(obs), process(next_obs)

        # transform obs
        latent_obs, latent_next_obs, indices_obs, indices_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward,
                              indices_obs, indices_next_obs)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward,
                                  indices_obs, indices_next_obs)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    tmp = RNNCatCell(latent_shape, code_dim, K, ASIZE, RSIZE)
    tmp.load_state_dict({k.strip('_l0'): v for k, v in mdrnn.state_dict().items()})
    save_checkpoint({
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, tmp, is_best, rnn_dir)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
