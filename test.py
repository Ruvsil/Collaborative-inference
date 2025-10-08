import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *
import pickle
import os


clients = {}
routing_nets = {}
for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
    routing_nets[i] = (routing_network(i, 10, INPUT_DIM, 25 , NUM_CLIENTS))

loss = torch.nn.CrossEntropyLoss()
routing_loss = torch.nn.CrossEntropyLoss()


class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))

for key in range(NUM_CLIENTS):
    clients[int(key)] = torch.load(f'./client_base_{int(key)}')
    routing_nets[int(key)] = torch.load(f'./routing_{int(key)}')



losssssss = []
with torch.no_grad():
    for x, y in test_loader:
        y_1hot = one_hot_encode(y, len(CLSS))
        x = torch.flatten(x, start_dim=1)
        o = clients[int(key)](x)
        #print(torch.argmax(o, dim=1), y_1hot)
        los = loss(o, y_1hot)
        losssssss.append(los)

print(sum(losssssss)/len(losssssss))