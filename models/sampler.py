import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class MiniSampler(nn.Module):
    def __init__(self, in_size: int, out_size, hidden_size=None, **kwargs):
        super().__init__()

        self.d1 = nn.Linear(in_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.d1(x)
        x = F.leaky_relu(x)
        return self.d2(x)


class Sampler(nn.Module):

    def __init__(self, latent_dim: int, device, hidden_size=20, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.mini_samplers = []
        for i in range(1, self.latent_dim + 1):  # First sample is not yet an output
            self.mini_samplers.append(MiniSampler(i, 2, hidden_size).to(device))

    #
    # def to(self, device):
    #     super(Sampler, self).to(device)
    #     for i in range(1, self.latent_dim + 1):
    #         self.mini_samplers[i].to(device)

    def forward(self, mu, log_var, sample=False):
        first_mu = mu[:, 0]
        first_std = torch.exp(0.5 * log_var[:, 0])
        eps = torch.randn_like(first_std)
        input = eps * first_std + first_mu
        input = input.unsqueeze(1)

        mus_resampled = []
        log_vars_resampled = []
        samples = []

        for idx in range(self.latent_dim):
            output = self.mini_samplers[idx](input)
            mu_resampled = output[:, 0]
            log_var_resampled = output[:, 1]
            eps = torch.randn_like(first_std)
            std_resampled = torch.exp(0.5 * log_var_resampled)
            new_input = eps * std_resampled + mu_resampled
            new_input = new_input.unsqueeze(1)
            mus_resampled.append(mu_resampled)
            log_vars_resampled.append(log_var_resampled)
            samples.append(new_input)
            if not sample:
                new_input = mu[:, idx].unsqueeze(1)
            input = torch.cat([input, new_input], 1)

        mus_resampled = torch.stack(mus_resampled, dim=1)
        log_vars_resampled = torch.stack(log_vars_resampled, dim=1)
        samples = torch.cat(samples, dim=1)

        return mus_resampled, log_vars_resampled, samples
