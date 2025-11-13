import os
import numpy as np
import random
from scipy.stats import entropy

from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
import torch

def get_params():
    dataset = 'cifar'
    if dataset == 'mnist':
        input_dim = 28 * 28
    else:
        input_dim = 32 * 32 * 3
    data_path = './datasets'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if dataset == 'mnist':
        ds_train = MNIST(data_path, train=True, download=True, transform=transform)
        ds_test = MNIST(data_path, train=False, download=True, transform=transform)
    if dataset == 'cifar':
        ds_train = CIFAR10(data_path, train=True, download=True, transform=transform)
        ds_test = CIFAR10(data_path, train=False, download=True, transform=transform)

    params_dict = {
        'input_dim' : input_dim,
        'dataset' : dataset,
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'ds_train' : ds_train,
        'ds_test' : ds_test,
        'test_loader' : DataLoader(ds_test, batch_size=128, shuffle=True),
        'clss' : ds_train.classes,
        'hidden_dim' : 2048,
        'rout_hidden_dim' : 2048,
        'output_dim' : len(ds_train.classes),
        'n_layers' : 3,
        'rout_n_layers' : 10,
        'num_clients' : 10,
        'entropy_threshold' : 0.3,
        'main_clss_percent' : 0.6,
        'client_train' : True,
    }

    # params_dict['ds_train'] = ds_train
    # params_dict['ds_test'] = ds_test
    # test_loader = DataLoader(ds_test, batch_size=128, shuffle=True)
    # params_dict['clss'] = ds_train.classes
    #
    # params_dict['hidden_dim'] = 2048
    # params_dict['output_dim'] = len(ds_train.classes)
    # params_dict['n_layers'] = 3
    #
    # params_dict['num_clients'] = 10
    # ENTROPY_TRSH = 0.3
    # client_train = True
    return params_dict