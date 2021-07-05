import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from data import CelebA
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np

from models import BaseVAE
from validation import Validator


class Trainer:

    def __init__(self,
                 vae_model: BaseVAE,
                 device,
                 experiment_name,
                 limit_data,
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
        self.validator = Validator(self.device, dataset=self.params["dataset"], dataloader=self.val_loader,
                                   score_model_device=self.device,
                                   stats_file_name=f"{self.params['dataset']}_seed_{self.params['manual_seed']}_limit_{int(limit_data * 100)}")
        self.save_dir = f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"

    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform, limit_data=self.limit_data)
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
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def sample_images(self, epoch):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.val_loader))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
                          f"recons_epoch_{epoch}.png",
                          normalize=True,
                          nrow=12)

        samples = self.model.sample(144, self.device, labels=test_label)
        vutils.save_image(samples.cpu().data,
                          f"{self.params['save_dir']}{self.experiment_name}/v_{self.version:03d}/"
                          f"samples_{epoch}.png",
                          normalize=True,
                          nrow=12)

        del test_input, recons  # , samples

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
            losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                real_img, labels = batch
                real_img = real_img.to(self.device)
                labels = labels.to(self.device)
                results = self.model.forward(real_img, labels=labels)
                train_loss = self.model.loss_function(*results,
                                                      M_N=self.params['batch_size'] / self.num_train_imgs,
                                                      batch_idx=batch_idx)
                loss = train_loss['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # if epoch % 2 == 1:
            print(f"Epoch: {epoch} mean loss {np.mean(losses)}")
            if epoch % self.params["save_interval"] == 0:
                fid, precision, recall, fid_r, precision_r, recall_r = self.validator.compute_fid(self.model)
                print(f"FID: {fid}, Precision: {precision}, Recall: {recall}")
                print(f"Reconstructions FID: {fid_r}, Precision: {precision_r}, Recall: {recall_r}")
                results_samples.append([fid, precision, recall])
                np.save(f"{self.save_dir}results_samples",
                        results_samples)
                results_recon.append([fid_r, precision_r, recall_r])
                np.save(f"{self.save_dir}results_recon",
                        results_recon)
                self.sample_images(epoch)
            if self.params['scheduler_gamma'] is not None:
                scheduler.step()
        torch.save(self.model, f"{self.save_dir}model")
