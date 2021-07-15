import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from data import CelebA
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np

from models import BaseVAE
from validation import Validator
import wandb


class Trainer:

    def __init__(self,
                 vae_model: BaseVAE,
                 device,
                 experiment_name,
                 limit_data,
                 skip_validation,
                 params: dict) -> None:
        # super(VAEXperiment, self).__init__()

        self.model = vae_model.to(device)
        all_params = {}
        for group in params:
            all_params.update(params[group])
        self.params = all_params
        self.limit_data = limit_data
        self.device = device
        self.experiment_name = experiment_name
        if os.path.exists(f"{self.params['save_dir']}{self.experiment_name}/"):
            self.version = len(next(os.walk(f"{self.params['save_dir']}{self.experiment_name}/"))[1])
        else:
            self.version = 0
        os.makedirs(f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}", exist_ok=True)
        self.num_train_imgs = 0
        self.num_val_imgs = 0
        self.val_loader = None
        self.train_loader = self.train_dataloader()
        self.init_val_dataloader()
        if skip_validation:
            self.validator = None
        else:
            self.validator = Validator(self.device, dataset=self.params["dataset"], dataloader=self.val_loader,
                                       score_model_device=self.device,
                                       stats_file_name=f"{self.params['dataset']}_seed_{self.params['manual_seed']}_limit_{int(limit_data * 100)}")
        self.save_dir = f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
        wandb.init(project="stepVAE", name=self.experiment_name, config=self.params)
        if not os.environ.get("WANDB_MODE") == "disabled":
            wandb.watch(self.model, log_freq=100)

        self.cov_matrix = None
        self.real_means = None
        self.skip_validation = skip_validation

    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform, limit_data=self.limit_data)
        elif self.params['dataset'] == 'cifar':
            dataset = CIFAR10(root=self.params['data_path'],
                              train=True,
                              download=True,
                              transform=transform)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    def init_val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.val_loader = DataLoader(CelebA(root=self.params['data_path'],
                                                split="test",
                                                transform=transform, limit_data=self.limit_data),
                                         batch_size=144,
                                         shuffle=True,
                                         drop_last=True)
            self.num_val_imgs = len(self.val_loader)
        elif self.params['dataset'] == 'cifar':
            self.val_loader = DataLoader(CIFAR10(root=self.params['data_path'],
                                                 train=False,
                                                 transform=transform,
                                                 download=True),
                                         batch_size=144,
                                         shuffle=True,
                                         drop_last=True)
            self.num_val_imgs = len(self.val_loader)
        else:
            raise ValueError('Undefined dataset type')

        return self.val_loader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'cifar':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def sample_images(self, epoch):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.val_loader))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)
        recons = self.model.generate(test_input, labels=test_label)
        recons_wandb = wandb.Image(recons[:16], caption="recons")
        wandb.log({"recons": recons_wandb})
        vutils.save_image(recons.data,
                          f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
                          f"recons_epoch_{epoch}.png",
                          normalize=True,
                          nrow=12)

        samples = self.model.sample(144, self.device, labels=test_label)
        samples_wandb = wandb.Image(samples[:16], caption="samples")
        wandb.log({"samples": samples_wandb})
        vutils.save_image(samples.cpu().data,
                          f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
                          f"samples_{epoch}.png",
                          normalize=True,
                          nrow=12)

        # samples_cov = self.model.sample(144, self.device, cov_matrix=self.cov_matrix, real_means=self.real_means,
        #                                 labels=test_label)
        # samples_cov_wandb = wandb.Image(samples_cov[:16], caption="samples")
        # wandb.log({"samples_cov": samples_cov_wandb})
        # vutils.save_image(samples_cov.cpu().data,
        #                   f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
        #                   f"samples_conv_{epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        del test_input, recons  # , samples

    def cov(self, X):
        X = X.rot90()
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def corr(self, X, eps=1e-08):
        X = X.rot90()
        D = X.shape[-1]
        std = torch.std(X, dim=-1).unsqueeze(-1)
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = (X - mean) / (std + eps)
        return 1 / (D - 1) * X @ X.transpose(-1, -2)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                         gamma=self.params['scheduler_gamma'])

        print(self.model)
        print("Training started")
        with open(f"{self.save_dir}/params.txt", "w") as text_file:
            text_file.write(str(self.params))
        results_recon = []
        results_samples = []
        for epoch in range(self.params['max_epochs']):
            mus = []
            losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                real_img, labels = batch
                real_img = real_img.to(self.device)
                labels = labels.to(self.device)
                results = self.model.forward(real_img, labels=labels)
                samples = results[4]
                mus.append(samples)
                # log_var = results[3]
                # std = torch.exp(0.5 * log_var)

                train_loss = self.model.loss_function(*results,
                                                      M_N=self.params['batch_size'] / self.num_train_imgs,
                                                      batch_idx=batch_idx)
                loss = train_loss['loss']
                wandb.log(train_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # if epoch % 2 == 1:
            print(f"Epoch: {epoch} mean loss {np.mean(losses)}")
            with torch.no_grad():
                mus = torch.cat(mus)
                corr_matrix = self.corr(mus)
                cov_matrix = self.cov(mus)
                real_means = torch.mean(mus, dim=0)
                real_stds = torch.std(mus, dim=0)
                self.real_means = real_means
                self.cov_matrix = cov_matrix
                corr_image = wandb.Image(corr_matrix)
            wandb.log({"corr_matrix": corr_image})
            cov_image = wandb.Image(cov_matrix)
            wandb.log({"cov_matrix": cov_image})
            if epoch % self.params["save_interval"] == 0:
                if not self.skip_validation:
                    fid, precision, recall, fid_r, precision_r, recall_r, fid_c, precision_c, recall_c = self.validator.compute_fid(
                        self.model, cov_matrix=None, real_means=real_means)
                    print(f"FID: {fid}, Precision: {precision}, Recall: {recall}")
                    print(f"Reconstructions FID: {fid_r}, Precision: {precision_r}, Recall: {recall_r}")
                    results_samples.append([fid, precision, recall])
                    np.save(f"{self.save_dir}results_samples",
                            results_samples)
                    results_recon.append([fid_r, precision_r, recall_r])
                    wandb.log({"fid_recon": fid_r,
                               "fid_sample": fid,
                               "fid_cov": fid_c,
                               "precision_recon": precision_r,
                               "precision_sample": precision,
                               "precision_cov": precision_c,
                               "recall_recon": recall_r,
                               "recall_sample": recall,
                               "recall_c": recall_c,
                               "real_means": self.real_means,
                               "real_stds": real_stds
                               })
                    np.save(f"{self.save_dir}results_recon",
                            results_recon)
                self.sample_images(epoch)
            if self.params['scheduler_gamma'] is not None:
                scheduler.step()
        torch.save(self.model, f"{self.save_dir}model")
