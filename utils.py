import os
import numpy as np
import random

from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
import torch

DATASET = 'mnist'

if DATASET == 'mnist':
    INPUT_DIM = 28*28
else:
    INPUT_DIM = 32*32*3



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

test_loader = DataLoader(ds_test, batch_size=128, shuffle=False)

CLSS = ds_train.classes

HIDDEN_DIM = 5
OUTPUT_DIM = len(CLSS)
N_LAYERS = 4

NUM_CLIENTS = 10


def create_mixed_datasets(class_datasets, num_of_clients, num_main_classes, rnd_ratio, datasets_len):
    ret = {}
    for i in range(num_of_clients):
        main_clss = [i]
        while len(main_clss) < num_main_classes:
            cls = np.random.randint(0, len(CLSS))
            if cls not in main_clss:
                main_clss.append(cls)
        base_ds = []
        for c in main_clss:
            base_ds.append(Subset(class_datasets[c], np.arange(int((rnd_ratio*datasets_len)/num_main_classes))))
        #base2 = Subset(class_datasets[cls2], np.arange(int((rnd_ratio*datasets_len)/num_main_classes)))
        base = base_ds[0]
        for b in base_ds[1:]:
            base += b
        rnd = []
        for key, value in class_datasets.items():
            if key not in main_clss:
                rnd = ConcatDataset([rnd, value])
        extra = Subset(rnd, random.sample(range(len(rnd)), int((1-rnd_ratio)*datasets_len)))
        ret[i] = (mixed_dataset(base, extra, datasets_len, main_clss))
    return ret



class mixed_dataset(Dataset):
    def __init__(self, base, extra, full_length, main_clss, transform=None):
        self.base = base
        self.extra = extra
        self.full_length = full_length
        self.main_clss = main_clss

    def __getitem__(self, index):
        if index < len(self.base):
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


# class routing_network(nn.Module):
#     def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
#         for lay in range(n_layers):
#             self.net.append(nn.Linear(hidden_dim, hidden_dim))
#             self.net.append(nn.ReLU())
#         self.net.append(nn.Linear(hidden_dim, output_dim))
#         # self.net.append(nn.Softmax())
#
#     def forward(self, input):
#         return self.net(input)


# def routing(net, clients, input, prediction, labels, main_clss_dict, loss, optim):
#     prediction = torch.argmax(prediction, dim=1)
#     mask = prediction == len(CLSS)
#     input_extracted = input[mask]
#     label_extracted = labels[mask]
#     y = []
#     for labl in label_extracted:
#         y.append(main_clss_dict[int(labl)][np.random.randint(len(main_clss_dict[int(labl)]))])
#
#     #print('aaa' , input_extracted, input_extracted.size())
#     o = net(input_extracted)
#
#     o_max = torch.argmax(o, dim=1)
#     routed = []
#
#     with torch.no_grad():
#         for idx, client_id in enumerate(o_max):
#             routed.append(clients[int(client_id)](input_extracted[idx]))
#         if routed:
#             routed = torch.stack(routed)
#     if len(routed):
#         y = one_hot_encode(y, len(CLSS))
#         # y = torch.tensor(y, dtype=torch.float64)
#         print(y)
#         print(o_max)
#         #print(o)
#         l = loss(o, y)
#         print('ext_loss', l)
#         l.backward()
#         optim.step()
#         optim.zero_grad()
#     return routed, mask




def one_hot_encode(batch, num_classes, main_clss=None):
    ret = []
    for y in batch:
        if main_clss:
            tensor = torch.zeros(num_classes+1)
        else:
            tensor = torch.zeros(num_classes)
        tensor[y] += 1
        if main_clss and y not in main_clss:
            tensor[-1] += 1
        ret.append(tensor)
    return torch.stack(ret)