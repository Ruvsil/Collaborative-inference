import os
import numpy as np
import random

from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler
import torch

DATASET = 'mnist'

if DATASET == 'mnist':
    INPUT_DIM = 28*28
else:
    INPUT_DIM = 32*32*3

HIDDEN_DIM = 3
OUTPUT_DIM = 10
N_LAYERS = 3

NUM_CLIENTS = 10

data_path = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

if not os.path.exists(data_path):
    os.mkdir(data_path)
if DATASET == 'mnist':
    ds_train = MNIST(data_path, train=True, download=True, transform=transform)
    ds_test = MNIST(data_path, train=False, download=True, transform=transform)
if DATASET == 'cifar':
    ds_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    ds_test = CIFAR10(data_path, train = False, download=True, transform=transform)

CLSS = ds_train.classes

def create_mixed_datasets(class_datasets, num_of_clients, rnd_ratio, datasets_len):
    ret = {}
    for i in range(num_of_clients):
        cls = np.random.randint(0,len(CLSS))
        base = Subset(class_datasets[cls], np.arange(int(rnd_ratio*datasets_len)))
        rnd = []
        for key, value in class_datasets.items():
            if key is not cls:
                rnd = ConcatDataset([rnd, value])
        extra =  Subset(rnd, random.sample(range(len(rnd)), datasets_len-len(base)))
        ret[cls] = (mixed_dataset(base,extra, datasets_len))
    return ret



class mixed_dataset(Dataset):
    def __init__(self, base, extra, full_length, transform=None):
        self.base = base
        self.extra = extra
        self.full_length = full_length
    def __getitem__(self, index):
        if  index< len(self.base):
            return self.base[index]
        else:
            return self.extra[len(self.base)-index]

    def __len__(self):
        return self.full_length

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

class client_network(nn.Module):
    def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.client_id = client_id
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        for lay in range(n_layers):
            self.net.append(nn.Linear(hidden_dim,hidden_dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dim, output_dim))
        #self.net.append(nn.Softmax())

    def forward(self, input):
        return self.net(input)

def one_hot_encode(index, num_classes):
    tensor = torch.zeros(num_classes)
    tensor[index] += 1
    return tensor
