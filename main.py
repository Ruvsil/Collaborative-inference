import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *
print(len(ds_train))
print(CLSS)
clients = {}

for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
#dataloader_train = DataLoader(ds_train, batch_size=32, shuffle=False)
#dataloader_test = DataLoader(ds_test)

clss_data = {}

for i,cls in enumerate(CLSS):
    clss_data[i] = []

loss = torch.nn.CrossEntropyLoss()


class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)



clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))
    print(f"Class {class_label}: {len(clss_data[class_label])} samples")

mixed_data = create_mixed_datasets(clss_data, 10, 1, 5000)

for id,dat in mixed_data.items():
    mixed_data[id] = DataLoader(dat, batch_size=128, shuffle=True)

for k,l in mixed_data.items():
    print('='*30)
    print(k)
    optim = torch.optim.SGD(clients[int(k)].parameters(), lr=0.2)
    for epoch in range(3):
        for x,y in l:

            x = torch.flatten(x, start_dim=1)
            o = clients[int(k)](x)
            los = loss(o,y)
            los.backward()
            optim.step()
            optim.zero_grad()
            print(los)
print(clss_data)