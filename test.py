import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *
import pickle
import os

clients = {}
routing_nets = {}


loss = torch.nn.CrossEntropyLoss()
routing_loss = torch.nn.CrossEntropyLoss()


class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))



losses = []

for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM+1))

for key in range(NUM_CLIENTS):
    clients[int(key)] = torch.load(f'./client_base_{int(key)}', weights_only=False)
with torch.no_grad():
    for key, client in clients.items():
        client_loss = []
        for x, y in test_loader:
            y_1hot = one_hot_encode(y, len(CLSS))
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)
            #print(torch.argmax(o, dim=1), y_1hot)
            #los = loss(o[:,:-1], y_1hot)
            los = loss(o, y_1hot)
            client_loss.append(los)
        print(key, sum(client_loss)/len(client_loss))
        losses.append(sum(client_loss)/len(client_loss))
print(losses)
print(sum(losses)/len(losses))

losses = []

for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM+1))
    routing_nets[i] = (routing_network(i, 10, INPUT_DIM, 25, NUM_CLIENTS))

for key in range(NUM_CLIENTS):
    clients[int(key)] = torch.load(f'./client_{int(key)}', weights_only=False)
    routing_nets[int(key)] = torch.load(f'./routing_{int(key)}', weights_only=False)
with torch.no_grad():
    for key, client in clients.items():
        client_loss = []
        for x, y in test_loader:
            y_1hot = one_hot_encode(y, len(CLSS))
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)

            o_max = torch.argmax(o, dim=1)
            mask = o_max == len(CLSS)
            extracted = x[mask]
            prediction=[]
            routed = torch.argmax(routing_nets[key](extracted), dim=1)
            if len(routed):
                for i,(sample, cli) in enumerate(zip(extracted, routed)):
                    prediction.append(clients[int(cli)](sample))
                prediction = torch.stack(prediction)
                routed = None
            # print(torch.argmax(o, dim=1), y_1hot)
                o[mask] = prediction

            los = loss(o[:,:-1], y_1hot)

            # print(torch.argmax(o[:,:-1],dim=1).float())
            # print(torch.argmax(y_1hot,dim=1).float())
            #print(loss(torch.argmax(o[:,:-1],dim=1).float(),torch.argmax(y_1hot,dim=1).float()))
            client_loss.append(los)
        print(key, sum(client_loss) / len(client_loss))
        losses.append(sum(client_loss) / len(client_loss))
print(losses)
print(sum(losses)/len(losses))