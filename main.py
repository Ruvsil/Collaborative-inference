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
routing_loss = torch.nn.L1Loss()
routing_net = routing_network(5, INPUT_DIM, 10, NUM_CLIENTS)
routing_optim = torch.optim.SGD(routing_net.parameters(), lr=0.2)

class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))
    print(f"Class {class_label}: {len(clss_data[class_label])} samples")

main_clss_dict = defaultdict(list)

mixed_data = create_mixed_datasets(clss_data, num_of_clients=NUM_CLIENTS, num_main_classes=2, rnd_ratio=0.7, datasets_len=10000)

for id, dat in mixed_data.items():
    for cls in dat.main_clss:
        main_clss_dict[cls].append(id)
    # for cls in range(len(CLSS)):
    #     if cls in dat.main_clss:
    #         main_clss_dict[cls].append(id)
    mixed_data[id] = (DataLoader(dat, batch_size=128, shuffle=True), dat.main_clss)
print(main_clss_dict)
for key, (loader, main_clss) in mixed_data.items():
    print('='*30)
    print(key)
    optim = torch.optim.SGD(clients[int(key)].parameters(), lr=0.2)
    for epoch in range(5):
        for x, y in loader:
            y_1hot = one_hot_encode(y, len(CLSS), main_clss)
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)
            routed, mask = routing(routing_net, clients, x, o, y, main_clss_dict, routing_loss, routing_optim)
            #print(torch.argmax(o, dim=1), y_1hot)
            los = loss(o,y_1hot)
            # if len(routed):
            #     los += loss(routed, y_1hot[mask])
            los.backward()
            optim.step()
            optim.zero_grad()
            print(los)