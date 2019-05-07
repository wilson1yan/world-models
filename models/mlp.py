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
