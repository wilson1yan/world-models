import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[],
                 output_activation=None):
        super(MLP, self).__init__()

        modules = []
        hprev = input_dim
        for h in hidden_dims + [output_dim]:
            modules.append(nn.Linear(hprev, h))
            modules.append(nn.ReLU())
            hprev = h
        modules.pop(-1)
        self.model = nn.Sequential(*modules)
        self.output_activation = output_activation

    def forward(self, x):
        out = self.model(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class SimpleConv(nn.Module):
    def __init__(self, in_channels, n_filters, n_layers, output_channels=None,
                 output_activation=None):
        super(SimpleConv, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.output_channels = output_channels
        self.output_activation = output_activation

        layers = []
        c = in_channels
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c, n_filters, 3, padding=1))
            layers.append(nn.ReLU())
            c = n_filters
        self.process = nn.Sequential(*layers)

        if output_channels is not None:
            self.output_layer = nn.Conv2d(c, output_channels)
        else:
            self.output_layer = nn.Linear(4*4*n_filters, 1)

    def forward(self, x):
        x = self.process(x)

        if self.output_channels is None:
            x = x.view(x.size(0), -1)
        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
