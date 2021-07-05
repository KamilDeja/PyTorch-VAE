import os

import torch
import numpy as np

from evaluation.fid import calculate_frechet_distance
from evaluation.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair


class Validator:
    def __init__(self, device, dataset, stats_file_name, dataloader, score_model_device=None):
        self.device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.dataloader = dataloader

        print("Preparing validator")
        if dataset in ["MNIST", "Omniglot"]:  # , "DoubleMNIST"]:
            if dataset in ["Omniglot"]:
                from evaluation_models.lenet_Omniglot import Model
            # elif dataset == "DoubleMNIST":
            #     from vae_experiments.evaluation_models.lenet_DoubleMNIST import Model
            else:
                from evaluation_models.lenet import Model
            net = Model()
            model_path = "vae_experiments/evaluation_models/lenet_" + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.dims = 128 if dataset in ["Omniglot", "DoubleMNIST"] else 84  # 128
            self.score_model_func = net.part_forward
        elif dataset.lower() in ["celeba", "doublemnist", "fashionmnist"]:
            from evaluation_models.inception import InceptionV3
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx])
            if score_model_device:
                model = model.to(score_model_device)
            model.eval()
            self.score_model_func = lambda batch: model(batch)[0]
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    def compute_fid(self, model):
        model.eval()
        test_loader = self.dataloader
        with torch.no_grad():
            distribution_orig = []
            distribution_gen = []
            distribution_rec = []

            precalculated_statistics = False
            os.makedirs(f"orig_stats/", exist_ok=True)
            stats_file_path = f"orig_stats/{self.dataset}_{self.stats_file_name}.npy"
            if os.path.exists(stats_file_path):
                print(f"Loading cached original data statistics from: {self.stats_file_name}")
                distribution_orig = np.load(stats_file_path)
                precalculated_statistics = True

            print("Calculating FID:")
            for idx, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                y = batch[1]

                example = model.sample(len(y), self.device, labels=y)
                res = model.forward(x, labels=y)
                recon = res[0]
                if not precalculated_statistics:
                    if self.dataset.lower() in ["fashionmnist", "doublemnist"]:
                        x = x.repeat([1, 3, 1, 1])
                    distribution_orig.append(self.score_model_func(x).cpu().detach().numpy())
                if self.dataset.lower() in ["fashionmnist", "doublemnist"]:
                    example = example.repeat([1, 3, 1, 1])
                    recon = recon.repeat([1, 3, 1, 1])
                distribution_gen.append(self.score_model_func(example))
                distribution_rec.append(self.score_model_func(recon))

            distribution_gen = torch.cat(distribution_gen).cpu().detach().numpy().reshape(-1, self.dims)
            distribution_rec = torch.cat(distribution_rec).cpu().detach().numpy().reshape(-1, self.dims)
            # distribution_gen = np.array(np.concatenate(distribution_gen)).reshape(-1, self.dims)
            if not precalculated_statistics:
                distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(-1, self.dims)
                np.save(stats_file_path, distribution_orig)

            precision, recall = compute_prd_from_embedding(
                eval_data=distribution_orig[np.random.choice(len(distribution_orig), len(distribution_gen), False)],
                ref_data=distribution_gen)
            precision, recall = prd_to_max_f_beta_pair(precision, recall)

            # recon
            precision_r, recall_r = compute_prd_from_embedding(
                eval_data=distribution_orig[np.random.choice(len(distribution_orig), len(distribution_rec), False)],
                ref_data=distribution_rec)
            precision_r, recall_r = prd_to_max_f_beta_pair(precision_r, recall_r)

            model.train()
            return calculate_frechet_distance(distribution_gen,
                                              distribution_orig), precision, recall, calculate_frechet_distance(
                distribution_rec, distribution_orig), precision_r, recall_r,
