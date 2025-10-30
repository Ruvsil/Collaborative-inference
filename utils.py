import os
import numpy as np
import random
from scipy.stats import entropy

from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATASET = 'mnist'

if DATASET == 'mnist':
    INPUT_DIM = 28 * 28
else:
    INPUT_DIM = 32 * 32 * 3

data_path = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

if not os.path.exists(data_path):
    os.mkdir(data_path)
if DATASET == 'mnist':
    ds_train = MNIST(data_path, train=True, download=True, transform=transform)
    ds_test = MNIST(data_path, train=False, download=True, transform=transform)
if DATASET == 'cifar':
    ds_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    ds_test = CIFAR10(data_path, train=False, download=True, transform=transform)

test_loader = DataLoader(ds_test, batch_size=128, shuffle=True)
CLSS = ds_train.classes

HIDDEN_DIM = 10
OUTPUT_DIM = len(CLSS)
N_LAYERS = 3

NUM_CLIENTS = 10
client_train = True


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
            base_ds.append(Subset(class_datasets[c], np.arange(int((rnd_ratio * datasets_len) / num_main_classes))))
        # base2 = Subset(class_datasets[cls2], np.arange(int((rnd_ratio*datasets_len)/num_main_classes)))
        base = base_ds[0]
        for b in base_ds[1:]:
            base += b
        rnd = []
        for key, value in class_datasets.items():
            if key not in main_clss:
                rnd = ConcatDataset([rnd, value])
        extra = Subset(rnd, random.sample(range(len(rnd)), int((1 - rnd_ratio) * datasets_len)))
        print(len(base), len(extra), datasets_len, main_clss)
        ret[i] = (mixed_dataset(base, extra, len(base) + len(extra), main_clss))
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
            # Corrected index for extra dataset access
            return self.extra[index - len(self.base)]

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
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dim, output_dim))
        # self.net.append(nn.Softmax())
        self.to(DEVICE)  # Move the entire network to the device

    def forward(self, input):
        return self.net(input)


class routing_network(nn.Module):
    def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.client_id = client_id
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        for lay in range(n_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dim, output_dim))
        # self.net.append(nn.Softmax())
        self.to(DEVICE)  # Move the entire network to the device

    def forward(self, input):
        return self.net(input)


def routing(net, clients, input, prediction, labels, main_clss_dict, loss, optim):
    prediction = torch.argmax(prediction, dim=1)
    mask = prediction == len(CLSS)
    input_extracted = input[mask].to(DEVICE)  # Move input to device
    label_extracted = labels[mask]
    y = []
    for labl in label_extracted:
        y.append(main_clss_dict[int(labl)][0])
    # print('aaa' , input_extracted, input_extracted.size())
    o = net(input_extracted)
    o_max = torch.argmax(o, dim=1)
    routed = []

    for idx, client_id in enumerate(o_max):
        routed.append(clients[int(client_id)](input_extracted[idx]))
    if routed:
        routed = torch.stack(routed)
    if len(routed):
        y = one_hot_encode(y, NUM_CLIENTS).to(DEVICE)  # Move target tensor to device
        # y = torch.tensor(y, dtype=torch.float64)
        #print(torch.argmax(y, dim=1))
        #print(o_max)
        #print('babababaababab', o.size(), y.size())
        l = loss(o, y)
        print('ext_loss', l)
        l.backward()
        optim.step()
        optim.zero_grad()
    return routed, mask


def entropy_routing(net, clients, input, prediction, labels, main_clss_dict, loss, optim):
    prediction = torch.softmax(prediction, dim=1)
    mask = torch.zeros(prediction.size()[0], device=DEVICE)  # Create mask on device
    for i, p in enumerate(prediction):
        entropy_value = entropy(p.detach().cpu().numpy())
        if entropy_value > 0.3:
            mask[i] += 1
    mask = mask.bool()
    input_extracted = input[mask].to(DEVICE)  # Move input to device
    label_extracted = labels[mask]
    y = []
    for labl in label_extracted:
        y.append(main_clss_dict[int(labl)][0])
    # print('aaa' , input_extracted, input_extracted.size())
    o = net(input_extracted)
    o_max = torch.argmax(o, dim=1)
    routed = []
    for idx, client_id in enumerate(o_max):
        routed.append(clients[int(client_id)](input_extracted[idx]))
    if routed:
        routed = torch.stack(routed)
    if len(routed):
        y = one_hot_encode(y, NUM_CLIENTS).to(DEVICE)  # Move target tensor to device
        # y = torch.tensor(y, dtype=torch.float64)
        #print(torch.argmax(y, dim=1))
        #print(o_max)
        #print('babababaababab', o.size(), y.size())
        l = loss(o, y)
        #print('ext_loss', l)
        l.backward()
        optim.step()
        optim.zero_grad()
    return routed, mask


def test_los(clients, routing_net, loss, key):
    with torch.no_grad():
        client_loss = []
        local_client = clients[int(key)]
        for x, y in test_loader:
            x = x.to(DEVICE)  # Move input to device
            y = y.to(DEVICE)  # Move labels to device
            y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
            x = torch.flatten(x, start_dim=1)
            o = local_client(x)

            o_max = torch.softmax(o, dim=1)

            mask = torch.zeros(o_max.size()[0], device=DEVICE)  # Create mask on device
            for i, p in enumerate(o_max):
                entropy_value = entropy(p.detach().cpu().numpy())
                if entropy_value > 0.3:
                    mask[i] += 1
            mask = mask.bool()
            extracted = x[mask]
            prediction = []
            if len(extracted) > 0:
                routed = torch.argmax(routing_net(extracted), dim=1)

                for i, (sample, cli) in enumerate(zip(extracted, routed)):
                    prediction.append(clients[int(cli)](sample))
                prediction = torch.stack(prediction)
                routed = None
                # print(torch.argmax(o, dim=1), y_1hot)
                o[mask] = prediction

            los = loss(o, y_1hot)

            # print(torch.argmax(o[:,:-1],dim=1).float())
            # print(torch.argmax(y_1hot,dim=1).float())
            # print(loss(torch.argmax(o[:,:-1],dim=1).float(),torch.argmax(y_1hot,dim=1).float()))
            client_loss.append(los)
        print(key, sum(client_loss) / len(client_loss))


def one_hot_encode(batch, num_classes, main_clss=None):
    ret = []
    for y in batch:
        if main_clss:
            tensor = torch.zeros(num_classes + 1)
        else:
            tensor = torch.zeros(num_classes)
        tensor[y] += 1
        if main_clss and y not in main_clss:
            tensor[-1] += 1
            # tensor[y] -= 1
        ret.append(tensor)
    return torch.stack(ret)