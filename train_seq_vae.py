import argparse
from os.path import join, exists
from os import makedirs
from functools import partial

from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models.seq_vae import SeqVAE
from utils.misc import save_checkpoint, IncreaseSize
from utils.misc import LSIZE, RED_SIZE, N_COLOR_DIM, ASIZE, RSIZE
from utils.metrics import compute_kl

from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutSequenceDataset

BSIZE = 16
SEQ_LEN = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process(obs):
    obs *= 255
    obs = torch.floor(obs / (2 ** 8 / N_COLOR_DIM)) / (N_COLOR_DIM - 1)
    return obs

def get_loss(obs, reward_target, done_target, recon_obs,
             obs_latents, obs_logsigmas,
             trans_latents, trans_logsigmas,
             rewards, terminals, include_reward, global_prior,
             model_name):
    """ Compute losses.

    :args obs: (BSIZE, SEQ_LEN, ASIZE, RED_SIZE, RED_SIZE) torch tensor (actual)
    :args recon_obs: (BSIZE, SEQ_LEN, ASIZE, RED_SIZE, RED_SIZE) torch tensor (reconstruction)
    :args obs_latents: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args obs_logsigmas: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args trans_latents: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args trans_logsigmas: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args rewards: (BSIZE, SEQ_LEN) torch tensor
    :args terminals: (BSIZE, SEQ_LEN) torch tensor

    :returns: dictionary of losses
    """
    if model_name == 'pixel_vae':
        target = (obs * (N_COLOR_DIM - 1)).long()
        recon_obs = torch.transpose(recon_obs, 1, 2)
        rcl = F.cross_entropy(recon_obs, target, reduce=False)
    else:
        rcl = (obs - recon_obs) ** 2
    rcl = rcl.view(rcl.size(0), -1)
    rcl = rcl.sum(-1).mean(0)
    kl = compute_kl(obs_latents, obs_logsigmas, trans_latents,
                    trans_logsigmas, time_dim=True)
    if global_prior:
        kl_global = compute_kl(obs_latents, obs_logsigmas,
                               torch.zeros_like(obs_latents).to(device),
                               torch.ones_like(obs_logsigmas).to(device),
                               time_dim=True)
    else:
        kl_global = torch.zeros(1).to(device)

    terminals, done_target = terminals.view(-1), done_target.view(-1)
    bce = F.binary_cross_entropy_with_logits(terminals, done_target)
    bce *= SEQ_LEN

    if include_reward:
        rew = ((rewards - reward_target) ** 2).mean() * SEQ_LEN
    else:
        rew = torch.zeros(1).to(device)

    loss = rcl + kl + bce + rew + kl_global

    return dict(loss=loss, rcl=rcl, kl=kl, bce=bce, rew=rew, kl_global=kl_global)

def data_pass(epoch, train, loader, seq_vae, optimizer, include_reward,
              global_prior, discrete, asize, model_name):
    if train:
        seq_vae.train()
    else:
        seq_vae.eval()

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_rcl = 0
    cum_kl = 0
    cum_bce = 0
    cum_rew = 0
    cum_kl_g = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, _ = [arr.to(device) for arr in data]
        obs = process(obs)

        if discrete:
            action = action.long().unsqueeze(-1)
            one_hot = torch.zeros(action.size(0), action.size(1), asize).to(device)
            one_hot.scatter_(2, action, 1)
            action = one_hot

        hiddens = [
            torch.zeros(obs.size(0), RSIZE).to(device)
            for _ in range(2)]

        if train:
            out = seq_vae(obs, action, hiddens)
            losses = get_loss(obs, reward, terminal, *out,
                              include_reward=include_reward,
                              global_prior=global_prior,
                              model_name=model_name)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = seq_vae(obs, action, hiddens)
                losses = get_loss(obs, reward, terminal, *out,
                                  include_reward=include_reward,
                                  global_prior=global_prior,
                                  model_name=model_name)
        cum_loss += losses['loss'].item()
        cum_rcl += losses['rcl'].item()
        cum_kl += losses['kl'].item()
        cum_bce += losses['bce'].item()
        cum_rew += losses['rew'].item()
        cum_kl_g += losses['kl_global'].item()

        pbar.set_postfix_str("loss={loss:10.6f} rcl={rcl:10.6f} "
                             "kl={kl:10.6f}, bce={bce:10.6f} "
                             "rew={rew:10.6f}, klg={klg:10.6f}".format(
                                 loss=cum_loss / (i + 1), rcl=cum_rcl / (i + 1),
                                 kl=cum_kl / (i + 1), bce=cum_bce / (i + 1),
                                 rew=cum_rew / (i + 1), klg=cum_kl_g / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--logdir', type=str, help='Directory where results are logged',
                        default='logs')
    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('--dataset', type=str, default='carracing')
    parser.add_argument('--include_reward', action='store_true',
                        help="Add a reward modelisation term to the loss.")
    parser.add_argument('--global_prior', action='store_true',
                        help="Add a global prior")
    parser.add_argument('--model', type=str, default='vae')
    args = parser.parse_args()

    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = True

    asize = 18
    discrete = True

    vae_dir = join(args.logdir, args.dataset, 'seq_vae')
    if not exists(vae_dir):
        makedirs(vae_dir)

    reload_file = join(vae_dir, 'best.tar')
    if not args.noreload and exists(reload_file):
        state = torch.load(reload_file)
        print("Reloading model at epoch {} "
              "with test error {}".format(
                  state['epoch'], state['precision']
              ))
        vae = torch.load(join(vae_dir, 'model_best.pt'))
        optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        earlystopping = EarlyStopping('min', patience=30)

        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        earlystopping.load_state_dict(state['earlystopping'])
    else:
        vae = SeqVAE(asize, args.model)
        optimizer = torch.optim.Adam(vae.parameters(), lr=3e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        earlystopping = EarlyStopping('min', patience=30)
    vae = vae.to(device)

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

    train = partial(data_pass, train=True, include_reward=args.include_reward,
                    global_prior=args.global_prior, loader=train_loader,
                    seq_vae=vae, optimizer=optimizer, discrete=discrete,
                    asize=asize, model_name=args.model)
    test = partial(data_pass, train=False, include_reward=args.include_reward,
                   global_prior=args.global_prior, loader=test_loader,
                   seq_vae=vae, optimizer=None, discrete=discrete,
                   asize=asize, model_name=args.model)

    cur_best = None
    for e in range(args.epochs):
        train(e)
        test_loss = test(e)
        scheduler.step(test_loss)
        earlystopping.step(test_loss)

        is_best = not cur_best or test_loss < cur_best
        if is_best:
            cur_best = test_loss

        save_checkpoint({
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": test_loss,
            "epoch": e}, vae, is_best, vae_dir)

        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(e))
            break

if __name__ == '__main__':
    main()
