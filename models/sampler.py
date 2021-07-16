import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Sampler(nn.Module):

    def __init__(self, latent_dim: int, hidden_size=20, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.lstm_unit = nn.LSTMCell(input_size=2, hidden_size=hidden_size)
        self.lstm_projection = nn.LSTMCell(input_size=2 * hidden_size, hidden_size=2)
        # self.lstm_sampler = nn.LSTM(input_size=2, hidden_size=15, proj_size=2, batch_first=True)

    def forward(self, mu, log_var, sample=False):
        first_mu = mu[:, 0]
        first_std = torch.exp(0.5 * log_var[:, 0])
        eps = torch.randn_like(first_std)
        input = eps * first_std + first_mu
        input = input.unsqueeze(1)
        input_indexes = torch.zeros_like(input)
        input = torch.cat([input, input_indexes], 1)
        # x = torch.stack([mu, std]).permute(2, 1, 0)  # @TODO Check and fix [batch_size, seq_len, input_size]
        # output, _ = self.lstm_sampler(x)
        hidden_state = torch.zeros(len(input), self.hidden_size).to(input.device)
        cell_state = torch.zeros(len(input), self.hidden_size).to(input.device)
        hidden_state_second = torch.zeros(len(input), 2).to(input.device)
        cell_state_second = torch.zeros(len(input), 2).to(input.device)

        mus_resampled = []
        log_vars_resampled = []
        samples = []

        for idx in range(self.latent_dim):
            hidden_state, cell_state = self.lstm_unit(input, (hidden_state, cell_state))
            output_first = torch.cat([hidden_state, cell_state], 1)
            # output_first = torch.tanh(output_first)
            output_first = F.leaky_relu(output_first)
            hidden_state_second, cell_state_second = self.lstm_projection(output_first,
                                                                          (hidden_state_second, cell_state_second))

            mu_resampled = hidden_state_second[:, 0]
            log_var_resampled = hidden_state_second[:, 1]
            eps = torch.randn_like(first_std)
            std_resampled = torch.exp(0.5 * log_var_resampled)
            input = eps * std_resampled + mu_resampled
            input = input.unsqueeze(1)
            mus_resampled.append(mu_resampled)
            log_vars_resampled.append(log_var_resampled)
            samples.append(input)
            if not sample:
                input = mu[:, idx].unsqueeze(1)

            input_indexes = torch.zeros_like(input) + idx
            input = torch.cat([input, input_indexes], 1)

        mus_resampled = torch.stack(mus_resampled, dim=1)
        log_vars_resampled = torch.stack(log_vars_resampled, dim=1)
        samples = torch.cat(samples, dim=1)

        return mus_resampled, log_vars_resampled, samples
