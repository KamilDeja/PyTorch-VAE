import torch
from models import BaseVAE
from models.sampler import Sampler
from torch import nn
from torch.nn import functional as F
from .types_ import *


class StepVanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(StepVanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())
        self.sampler = Sampler(latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """

        # eps = torch.randn_like(logvar)
        mus_resampled, log_var_resampled, samples = self.sampler(mu, logvar)
        # std = torch.exp(0.5 * logvar)
        return mus_resampled, log_var_resampled, samples

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        mu_repam, log_var_repam, z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z, mu_repam, log_var_repam]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        mu_repam = args[5]
        log_var_repam = args[6]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        mse_latent_mu_loss = F.mse_loss(mu, mu_repam)
        mse_latent_std_loss = F.mse_loss(log_var, log_var_repam)

        kld_loss_pre = torch.mean(-0.5 * torch.sum(1 + log_var[0:1] - mu[0:1] ** 2 - log_var[0:1].exp(), dim=1), dim=0)
        kld_loss_post = torch.mean(
            -0.5 * torch.sum(1 + log_var_repam[0:1] - mu_repam[0:1] ** 2 - log_var_repam[0:1].exp(), dim=1), dim=0)

        loss = 50 * recons_loss + kld_weight * (
                kld_loss_pre + kld_loss_post) + mse_latent_mu_loss + mse_latent_std_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss,
                'KLD': + kld_weight * (kld_loss_pre + kld_loss_post),
                'MSE_mu': mse_latent_mu_loss, 'MSE_std': mse_latent_std_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        cov_matrix = kwargs.get("cov_matrix")
        if cov_matrix != None:
            real_means = kwargs.get("real_means")
            m = torch.distributions.MultivariateNormal(real_means.to(current_device),
                                                       covariance_matrix=cov_matrix)
            z = m.sample_n(num_samples)
        else:
            mu = torch.zeros(num_samples, self.latent_dim).to(current_device)
            std = torch.ones(num_samples, self.latent_dim).to(current_device)
            mu, std, z = self.sampler(mu, std, sample=True)
            # z = z * std + mu

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
