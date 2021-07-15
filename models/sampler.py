import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Sampler(nn.Module):

    def __init__(self, latent_dim: int, hidden_size=20, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.lstm_unit = nn.LSTMCell(input_size=latent_dim - 1, hidden_size=hidden_size)
        self.lstm_projection = nn.LSTMCell(input_size=2 * hidden_size, hidden_size=2)
        # self.lstm_sampler = nn.LSTM(input_size=2, hidden_size=15, proj_size=2, batch_first=True)

    def forward(self, mu, log_var, sample=False):
        first_mu = mu[:, 0]
        first_std = torch.exp(0.5 * log_var[:, 0])
        eps = torch.randn_like(first_std)
        input = torch.zeros([first_mu.size(0), self.latent_dim - 1], requires_grad=False).to(first_mu.device)
        first_input = eps * first_std + first_mu
        input[:, 0] = first_input
        # x = torch.stack([mu, std]).permute(2, 1, 0)
        # output, _ = self.lstm_sampler(x)
        hidden_state = torch.zeros(len(input), self.hidden_size).to(input.device)
        cell_state = torch.zeros(len(input), self.hidden_size).to(input.device)
        hidden_state_second = torch.zeros(len(input), 2).to(input.device)
        cell_state_second = torch.zeros(len(input), 2).to(input.device)

        mus_resampled = []
        log_vars_resampled = []
        samples = []
        mus_resampled.append(torch.zeros_like(first_mu))
        log_vars_resampled.append(torch.ones_like(first_mu))  # @TODO:fix
        samples.append(first_input)

        for idx in range(1, self.latent_dim):
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
            temp_input = eps * std_resampled + mu_resampled
            if idx < self.latent_dim - 1:
                if sample:
                    input = torch.cat([input[:, :idx], temp_input.unsqueeze(1),
                                       torch.zeros([temp_input.size(0), self.latent_dim - idx - 2]).to(mu.device)], 1)
                # input[:, idx] = temp_input
                else:
                    input = torch.cat(
                        [mu[:, :idx + 1], torch.zeros([temp_input.size(0), self.latent_dim - idx - 2]).to(mu.device)], 1)
            # input = input.unsqueeze(1)
            mus_resampled.append(mu_resampled)
            log_vars_resampled.append(log_var_resampled)
            samples.append(temp_input)

        mus_resampled = torch.stack(mus_resampled, dim=1)
        log_vars_resampled = torch.stack(log_vars_resampled, dim=1)
        samples = torch.stack(samples, dim=1)

        return mus_resampled, log_vars_resampled, samples
