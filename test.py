import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *  # DEVICE is imported from utils
from config import *
import pickle
import os

clients = {}
routing_nets = {}

loss = torch.nn.CrossEntropyLoss().to(DEVICE)  # Move loss to device
routing_loss = torch.nn.CrossEntropyLoss().to(DEVICE)  # Move loss to device

class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))

losses = []

# Initialize networks and load state dicts
for i in range(NUM_CLIENTS):
    # client_network moves itself to DEVICE in its __init__
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
    # Load state dict
    # Using map_location to ensure the loaded model is on the current device
    clients[int(i)].load_state_dict(torch.load(f'./client_{int(i)}.pt', map_location=DEVICE, weights_only=True))

with torch.no_grad():
    for key, client in clients.items():
        client_loss = []
        for x, y in test_loader:
            x = x.to(DEVICE)  # Move input to device
            y = y.to(DEVICE)  # Move labels to device
            y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)

            # print(torch.argmax(o, dim=1), y_1hot)
            # los = loss(o[:,:-1], y_1hot)
            los = loss(o, y_1hot)
            client_loss.append(los)
        # Move loss to CPU for numpy sum/average calculation (optional but safer)
        print(key, sum(l.item() for l in client_loss) / len(client_loss))
        losses.append(sum(l.item() for l in client_loss) / len(client_loss))
print(losses)
print(sum(losses) / len(losses))

losses = []
accs = []
# Re-initialize networks and load state dicts
for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
    routing_nets[i] = (routing_network(i, 10, INPUT_DIM, 2048, NUM_CLIENTS))

for key in range(NUM_CLIENTS):
    # Load state dicts
    clients[int(key)].load_state_dict(torch.load(f'./client_{int(key)}.pt', map_location=DEVICE, weights_only=True))
    routing_nets[int(key)].load_state_dict(torch.load(f'./routing_{int(key)}.pt', map_location=DEVICE, weights_only=True))

with torch.no_grad():
    for key, client in clients.items():
        test_los(clients, routing_nets[int(key)], loss, key)
        client_loss = []
        client_acc = []
        for x, y in test_loader:
            x = x.to(DEVICE)  # Move input to device
            y = y.to(DEVICE)  # Move labels to device
            y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)

            o_max = torch.softmax(o, dim=1)

            mask = torch.zeros(o_max.size()[0], device=DEVICE)  # Create mask on device
            for i, p in enumerate(o_max):
                entropy_value = entropy(p.detach().cpu().numpy())
                if entropy_value > ENTROPY_TRSH:
                    mask[i] += 1
            mask = mask.bool()

            extracted = x[mask]
            prediction = []
            if len(extracted) > 0:
                routed = torch.argmax(routing_nets[key](extracted), dim=1)

                for i, (sample, cli) in enumerate(zip(extracted, routed)):
                    prediction.append(clients[int(cli)](sample))
                prediction = torch.stack(prediction)
                routed = None
                # print(torch.argmax(o, dim=1), y_1hot)
                o[mask] = prediction

            los = loss(o, y_1hot)
            acc = accuracy(o, y_1hot)
            # print(torch.argmax(o[:,:-1],dim=1).float())
            # print(torch.argmax(y_1hot,dim=1).float())
            # print(loss(torch.argmax(o[:,:-1],dim=1).float(),torch.argmax(y_1hot,dim=1).float()))
            client_loss.append(los)
            client_acc.append(acc)
        # Move loss to CPU for numpy sum/average calculation (optional but safer)
        print(key, sum(l.item() for l in client_loss) / len(client_loss))
        losses.append(sum(l.item() for l in client_loss) / len(client_loss))
        print(key, sum(client_acc) / len(client_acc))
        accs.append(sum(client_acc) / len(client_acc))
print(losses)
print(sum(losses) / len(losses))
print(accs)
print(sum(accs) / len(accs))