import os

# import torchcde
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as dto

# from TorchDiffEqPack.odesolver import ode_solver
import flow as flows
import sim_config
from global_config import DTYPE, get_device

class GaussianReparam:
    """Independent Gaussian posterior with re-parameterization trick."""

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def log_density(mu, log_var, z):
        n = dist.normal.Normal(mu, torch.exp(0.5 * log_var))
        log_p = torch.sum(n.log_prob(z), dim=-1)
        return log_p


# The same encoder structure
class EncoderLSTM(nn.Module, GaussianReparam):
    """Is used in the simulation setting. """
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True, device=None):
        # output dim is the dim of initial condition
        # input dim is observation and action

        super(EncoderLSTM, self).__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.model_name = "LSTMEncoder"

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        # The linear layer that maps from hidden state space to output space: predict mean
        self.lin = nn.Linear(hidden_dim, output_dim).to(self.device)

        self.log_var = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, x, a, mask):
        # y and t are the first k observations

        t_max = x.shape[0]

        x = x.squeeze()

        y_in = torch.cat([x, a], dim=-1) # concatenate observation and action
        mask_in = torch.cat([mask, torch.ones_like(a)], dim=-1)

        hidden = None

        for t in reversed(range(t_max)):
            obs = y_in[t : t + 1, ...] * mask_in[t : t + 1, ...]
            out, hidden = self.lstm(obs, hidden) 
            
        out_linear = self.lin(out)
        log_var = self.log_var(out)

        # B, D
        mu = out_linear[0, ...]
        log_var = log_var[0, ...]

        if self.normalize:
            # scale mu
            mu = torch.exp(mu) / 10
            # mask = torch.zeros_like(mu)
            # mask[:, 0] = 1
            # mu = mu * mask

            # scale var
            log_var = log_var - 5.0

        return mu, log_var