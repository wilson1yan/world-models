import torch

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)

    x = x.view(x_size, -1)
    y = y.view(y_size, -1)

    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def compute_kl(dist1_mu, dist1_logsigma, dist2_mu, dist2_logsigma, time_dim=False):
    kl = 2 * (dist2_logsigma - dist1_logsigma) + (2*dist1_logsigma).exp() / (2*dist2_logsigma).exp()
    kl += (dist2_mu - dist1_mu) ** 2 * (-2*dist2_logsigma).exp() - 1
    kl *= 0.5
    kl = kl.sum(-1)

    if time_dim:
        kl = kl.sum(1)

    return kl.mean(0)
