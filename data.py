import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FastCelebA(Dataset):
    def __init__(self, data, attr):
        self.dataset = data
        self.attr = attr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.attr[index]


def CelebA(root, transform, split="train", target_type='attr', limit_data=1):
    dataset = torchvision.datasets.CelebA(root=root, download=True, transform=transform, split=split,
                                          target_type=target_type)
    print("Loading data")
    save_path = f"{root}/fast_celeba_{split}_limit_{int(limit_data*100)}"
    if os.path.exists(save_path):
        fast_celeba = torch.load(save_path)
    else:
        train_loader = DataLoader(dataset, batch_size=int(len(dataset) * limit_data))
        data = next(iter(train_loader))
        fast_celeba = FastCelebA(data[0], data[1])
        torch.save(fast_celeba, save_path)
    return fast_celeba
