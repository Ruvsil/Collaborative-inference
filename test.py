import numpy as np
import torch
import json
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *  # DEVICE is imported from utils
from config import *
import pickle
import os

np.set_printoptions(suppress=True, precision=2)

experiment = 'prova01'
exp_dir = os.path.join(os.getcwd(), 'experiments', experiment)

with open(os.path.join(exp_dir, 'configuration.json'), 'r') as f:
    config = json.load(f)

clients = {}
clients_base = {}
routing_nets = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLSS = config['clss']
#
# for i in range(config['num_clients']):
#     # client_network and routing_network move themselves to DEVICE in their __init__
#     clients_base[i] = (client_network(i, config['n_layers'], config['input_dim'], config['hidden_dim'], config['output_dim'], DEVICE=DEVICE))
#     clients[i] = (client_network(i, config['n_layers'], config['input_dim'], config['hidden_dim'], config['output_dim'], DEVICE=DEVICE))
#     routing_nets[i] = (routing_network(i, config['route_n_layers'], config['input_dim'], config['route_hidden_dim'], config['num_clients'], DEVICE=DEVICE))
#     clients[int(i)].load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'client_{int(i)}.pt'), map_location=DEVICE, weights_only=True))
#     clients_base[int(i)].load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'client_base_{int(i)}.pt'), map_location=DEVICE, weights_only=True))
#
# loss = torch.nn.CrossEntropyLoss().to(DEVICE)  # Move loss function to device
# routing_loss = torch.nn.CrossEntropyLoss().to(DEVICE)  #

data_path = './datasets'

if not os.path.exists(data_path):
    os.mkdir(data_path)
if config['dataset'] == 'mnist':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds_train = MNIST(data_path, train=True, download=True, transform=transform)
    ds_test = MNIST(data_path, train=False, download=True, transform=transform)
if config['dataset'] == 'cifar':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    ds_test = CIFAR10(data_path, train=False, download=True, transform=transform)

test_loader = DataLoader(ds_test, batch_size=2048, shuffle=True)

data_dict = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'ds_train': ds_train,
    'ds_test': ds_test,
    'test_loader': test_loader,
}

test(exp_dir, config, data_dict)