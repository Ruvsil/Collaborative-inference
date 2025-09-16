import os
import numpy as np
import random
import pickle

from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler
import torch

DATASET = 'mnist'

if DATASET == 'mnist':
    INPUT_DIM = 28 * 28
else:
    INPUT_DIM = 32 * 32 * 3

data_path = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

if not os.path.exists(data_path):
    os.mkdir(data_path)
if DATASET == 'mnist':
    ds_train = MNIST(data_path, train=True, download=True, transform=transform)
    ds_test = MNIST(data_path, train=False, download=True, transform=transform)
if DATASET == 'cifar':
    ds_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    ds_test = CIFAR10(data_path, train=False, download=True, transform=transform)

CLSS = ds_train.classes

HIDDEN_DIM = 5
OUTPUT_DIM = len(CLSS)
N_LAYERS = 4

NUM_CLIENTS = 10


class MyDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)


def create_mixed_datasets(class_datasets, num_of_clients, num_main_classes, rnd_ratio, datasets_len):
    clients_datasets = {}
    ds_list = list(class_datasets.values())

    for i in range(num_of_clients):
        chosen_main_classes = np.random.choice(len(CLSS), num_main_classes, replace=False)
        main_sets = [class_datasets[c] for c in chosen_main_classes]
        main_set_size = int(datasets_len * (1 - rnd_ratio) / num_main_classes)

        main_subsets = [Subset(ds, np.random.choice(len(ds), main_set_size, replace=False)) for ds in main_sets]

        all_other_classes = [c for c in range(len(CLSS)) if c not in chosen_main_classes]
        rnd_set = ConcatDataset([class_datasets[c] for c in all_other_classes])

        rnd_set_size = int(datasets_len * rnd_ratio)
        if len(rnd_set) > 0:
            rnd_sub_size = min(len(rnd_set), rnd_set_size)
            rnd_sub = Subset(rnd_set, np.random.choice(len(rnd_set), rnd_sub_size, replace=False))
        else:
            rnd_sub = []

        combined_dataset = ConcatDataset(main_subsets + [rnd_sub])

        if len(combined_dataset) > datasets_len:
            sampler = RandomSampler(combined_dataset, num_samples=datasets_len, replacement=False)
            subset = MyDataset(Subset(combined_dataset, list(sampler)))
        else:
            subset = MyDataset(combined_dataset)

        subset.main_clss = chosen_main_classes
        clients_datasets[i] = subset

    return clients_datasets


class client_network(nn.Module):
    def __init__(self, id, n_layers, input_dim, hidden_dim, output_dim):
        super(client_network, self).__init__()
        self.id = id
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def one_hot_encode(labels, num_classes):
    """
    Converts a list of class labels to a one-hot encoded tensor.
    """
    tensor = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        tensor[i][label] = 1
    return tensor