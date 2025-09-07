import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *


clients = {}

for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM+1))

clss_data = {}

for i,cls in enumerate(CLSS):
    clss_data[i] = []

loss = torch.nn.CrossEntropyLoss()
routing_net = routing_network(3, INPUT_DIM, HIDDEN_DIM, NUM_CLIENTS)

class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))
    print(f"Class {class_label}: {len(clss_data[class_label])} samples")

mixed_data = create_mixed_datasets(clss_data, num_of_clients=NUM_CLIENTS, num_main_classes=2, rnd_ratio=0.7, datasets_len=10000)

for id, dat in mixed_data.items():
    mixed_data[id] = (DataLoader(dat, batch_size=128, shuffle=True), dat.main_clss)

for key, (loader, main_clss) in mixed_data.items():
    print('='*30)
    print(key)
    optim = torch.optim.SGD(clients[int(key)].parameters(), lr=0.2)
    for epoch in range(5):
        for x, y in loader:
            y = one_hot_encode(y, len(CLSS), main_clss)
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)
            routing(routing_net, clients, x, o)
            print(torch.argmax(o, dim=1), y)
            los = loss(o,y)
            los.backward()
            optim.step()
            optim.zero_grad()
            print(los)
print(clss_data)