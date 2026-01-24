import os
import numpy as np
import random
from scipy.stats import entropy
from config import *
from torch import nn
from torch.nn.functional import one_hot
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, Subset, ConcatDataset, RandomSampler, DataLoader
from collections import defaultdict

import torch


def create_mixed_datasets(class_datasets, num_of_clients, num_main_classes, rnd_ratio, datasets_len, CLSS):
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
        base = ConcatDataset([b for b in base_ds])
        # for b in base_ds[1:]:
        #     #base += b
        #     base = ConcatDataset([base, b])
        rnd = []
        for key, value in class_datasets.items():
            if key not in main_clss:
                rnd = ConcatDataset([rnd, value])

        extra = Subset(rnd, random.sample(range(len(rnd)), int((1 - rnd_ratio) * datasets_len)+1))
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

class client_network_cnn(nn.Module):
    def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim, backbone, DEVICE):
        super().__init__()
        self.client_id = client_id

        self.cnn = backbone # nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )

        self.net = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        # self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3), padding=1), nn.BatchNorm2d(32), nn.MaxPool2d((2,2)))
        # self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1), nn.BatchNorm2d(64), nn.MaxPool2d((2,2)))
        # self.cnn3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1), nn.BatchNorm2d(64), nn.MaxPool2d((2,2)))
        # self.cnn = nn.Sequential(self.cnn1,nn.ReLU(),self.cnn2,nn.ReLU(),self.cnn3)
        # self.net = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU())
        # for lay in range(n_layers):
        #     self.net.append(nn.Linear(hidden_dim, hidden_dim))
        #     self.net.append(nn.ReLU())
        # self.net.append(nn.Linear(hidden_dim, output_dim))
        # self.net.append(nn.Softmax())
        self.to(DEVICE)

    def encode(self, input):
        return self.cnn(input)

    def forward(self, input):
        return self.net(input)

# class client_network(nn.Module):
#     def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim, DEVICE):
#         super().__init__()
#         self.client_id = client_id
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
#         for lay in range(n_layers):
#             self.net.append(nn.Linear(hidden_dim, hidden_dim))
#             self.net.append(nn.ReLU())
#         self.net.append(nn.Linear(hidden_dim, output_dim))
#         # self.net.append(nn.Softmax())
#         self.to(DEVICE)
#
#     def forward(self, input):
#         return self.net(input)
#


class routing_network(nn.Module):
    def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim, backbone, DEVICE):
        super().__init__()
        # Use a simple CNN for routing to recognize features
        self.features = backbone

        # 32 channels * 8 * 8 spatial = 2048
        flat_size = 32 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.to(DEVICE)

    def forward(self, input):
        x = self.features(input)
        return self.classifier(x)

class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.feature_extractor(x)

# class routing_network(nn.Module):
#     def __init__(self, client_id, n_layers, input_dim, hidden_dim, output_dim, DEVICE):
#         super().__init__()
#         self.client_id = client_id
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
#         for lay in range(n_layers):
#             self.net.append(nn.Linear(hidden_dim, hidden_dim))
#             self.net.append(nn.ReLU())
#         self.net.append(nn.Linear(hidden_dim, output_dim))
#         # self.net.append(nn.Softmax())
#         self.to(DEVICE)  # Move the entire network to the device
#
#     def forward(self, input):
#         return self.net(input)


# def routing(net, clients, input, prediction, labels, main_clss_dict, loss, optim):
#     prediction = torch.argmax(prediction, dim=1)
#     mask = prediction == len(CLSS)
#     input_extracted = input[mask].to(DEVICE)  # Move input to device
#     label_extracted = labels[mask]
#     y = []
#     print(main_clss_dict)
#     for labl in label_extracted:
#         y.append(main_clss_dict[int(labl)][0])
#     # print('aaa' , input_extracted, input_extracted.size())
#     o = net(input_extracted)
#     o_max = torch.argmax(o, dim=1)
#     routed = []
#
#     for idx, client_id in enumerate(o_max):
#         routed.append(clients[int(client_id)](input_extracted[idx]))
#     if routed:
#         routed = torch.stack(routed)
#     if len(routed):
#         y = one_hot_encode(y, NUM_CLIENTS).to(DEVICE)  # Move target tensor to device
#         # y = torch.tensor(y, dtype=torch.float64)
#         #print(torch.argmax(y, dim=1))
#         #print(o_max)
#         #print('babababaababab', o.size(), y.size())
#         l = loss(o, y)
#         print('ext_loss', l)
#         l.backward()
#         optim.step()
#         optim.zero_grad()
#     return routed, mask


def entropy_routing_train(net, clients, input, encoded_input, prediction, labels, main_clss_dict, loss, optim, DEVICE, entropy_trsh, num_clients):
    o = net(input)
    clients_weights = torch.softmax(o, dim=1)

    routed_outputs = []

    for client_idx, client_model in clients.items():
        # (batch_size, output_dim)

        client_out = client_model(encoded_input)
        # (batch_size)
        specific_client_weights = clients_weights[:, client_idx]
        # (batch_size, 1)
        specific_client_weights = specific_client_weights.unsqueeze(1)
        weighted_out = client_out * specific_client_weights
        routed_outputs.append(weighted_out)

    if routed_outputs:
        # (num_Clients, batch_Size, output_dim)
        routed_tensor = torch.stack(routed_outputs)
        routed = torch.sum(routed_tensor, dim=0)

    return routed


    # #prediction = torch.softmax(prediction, dim=1)
    # #print('prediction',prediction)
    # #log_o_max = torch.log(prediction)
    # #print('input_shape', input.shape)
    # #print('encoded_shape', encoded_input.shape)
    # print(list(net.parameters())[0])
    # o = net(input)
    # #print(list(net.net[0].parameters()))
    # clients_weights = torch.softmax(o, dim=1)
    # #print('clients_weights', clients_weights.shape)
    # routed = []
    # for idx, client in clients.items():
    #     classes_probabilities = (client(encoded_input)*clients_weights[:,idx])
    #     #print(classes_probabilities.shape, clients_weights.shape)
    #     #print((classes_probabilities*clients_weights[idx]))
    #     routed.append(classes_probabilities)
    # if routed:
    #     routed = torch.stack(routed)
    #     routed = torch.mean(routed, dim=0)
    # #print('routed',routed)
    # #print('routed_shape',routed.shape)

    #return routed

def entropy_routing(net, clients, input, encoded_input, prediction, labels, main_clss_dict, loss, optim, DEVICE, entropy_trsh, num_clients):
    prediction = torch.softmax(prediction, dim=1)
    print('prediction',prediction)
    log_o_max = torch.log(prediction)
    print('log_o_max',log_o_max)
    entropy_values = -torch.sum(prediction * log_o_max, dim=1)
    mask = entropy_values > entropy_trsh
    input_extracted = input[mask].to(DEVICE)  # Move input to device
    encoded_extracted = encoded_input[mask].to(DEVICE)
    label_extracted = labels[mask]
    print('entropy', entropy_values)
    print('entropy_mean',sum(entropy_values)/len(entropy_values))
    print('routed_percentage',len(input_extracted)/len(prediction))
    y = []
    #print(main_clss_dict)
    for labl in label_extracted:
        if main_clss_dict[int(labl)]:
            y.append(main_clss_dict[int(labl)][0])
        else:
            y.append(0)
    # print('aaa' , input_extracted, input_extracted.size())
    o = net(input_extracted)
    #print(list(net.net[0].parameters()))
    o_max = torch.argmax(o, dim=1)
    routed = []
    for idx, client_id in enumerate(o_max):
        routed.append(clients[int(client_id)](encoded_extracted[idx]))
    if routed:
        routed = torch.stack(routed)
    if len(routed):
        y = one_hot_encode(y, num_clients).to(DEVICE)  # Move target tensor to device
        # y = torch.tensor(y, dtype=torch.float64)
        #print(torch.argmax(y, dim=1))
        #print(o_max)
        #print('babababaababab', o.size(), y.size())
        l = loss(o, y)
        #print('ext_loss', l)
        l.backward()
        optim.step()
        optim.zero_grad()
        #print(list(net.net[0].parameters()))
        print('routing_accuracy: ', accuracy(o,y))
        print('routing_loss: ', l)
    return routed, mask

def accuracy(prediction, target):
    prediction = torch.argmax(prediction, dim=1)
    correct = 0
    for p,t in zip(prediction, target):
        if t[p] == 1:
            correct += 1
    return correct/len(prediction)


def test_los(clients, routing_net, key, test_loader, DEVICE, CLSS, entropy_trsh, run):
    with torch.no_grad():
        conf_mat = np.zeros((len(CLSS), len(CLSS)))
        client_loss = []
        client_acc = []
        local_client = clients[int(key)]
        for x, y in test_loader:
            x = x.to(DEVICE)  # Move input to device
            y = y.to(DEVICE)  # Move labels to device
            y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
            encoded = clients[int(key)].encode(x)
            #encoded = torch.flatten(encoded, start_dim=1)
            #x = torch.flatten(x, start_dim=1)
            o = local_client(encoded)

            o_max = torch.softmax(o, dim=1)

            log_o_max = torch.log(o_max)
            entropy_values = -torch.sum(o_max * log_o_max, dim=1)
            mask = entropy_values > entropy_trsh
            entropy_min = min(entropy_values)
            entropy_max = max(entropy_values)
            entropy_mean =sum(entropy_values)/len(entropy_values)
            run.log({f'client{key}/entropy_min' : entropy_min,
                     f'client{key}/entropy_mean': entropy_mean,
                     f'client{key}/entropy_max' : entropy_max})
            encoded_extracted = encoded[mask]
            extracted = x[mask]
            prediction = []
            if len(extracted) > 0:
                routed = torch.argmax(routing_net(extracted), dim=1)

                for i, (sample, cli) in enumerate(zip(encoded_extracted, routed)):
                    prediction.append(clients[int(cli)](sample))
                prediction = torch.stack(prediction)
                routed = None
                # print(torch.argmax(o, dim=1), y_1hot)
                o[mask] = prediction
            for xo, yo in zip(o, y_1hot):
                conf_mat[torch.argmax(yo)][torch.argmax(xo)] += 1

            row_sums = conf_mat.sum(axis=1)
            row_sums_reshaped = row_sums[:, np.newaxis]
            conf_mat = conf_mat / row_sums_reshaped

            #los = loss(o, y_1hot)
            acc = accuracy(o, y_1hot)
            #f1 = F1Score(o, y_1hot)
            # print(torch.argmax(o[:,:-1],dim=1).float())
            # print(torch.argmax(y_1hot,dim=1).float())
            # print(loss(torch.argmax(o[:,:-1],dim=1).float(),torch.argmax(y_1hot,dim=1).float()))
            #client_loss.append(los)
            client_acc.append(acc)
        #print(key, 'loss', sum(client_loss) / len(client_loss))
        routing_ratio = sum(mask) / len(x)
        run.log({f'client{key}/routing_ratio' : routing_ratio})
        print('routed ratio', routing_ratio )
        mean_acc = sum(client_acc)/len(client_acc)
        print(key, 'acc', mean_acc)
        print(conf_mat)
        return mean_acc, conf_mat


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


def test(exp_dir, param, data):
    clients = {}
    routing_nets = {}
    conf_mat = np.zeros((len(param['clss']), len(param['clss'])))

    loss = torch.nn.CrossEntropyLoss().to(data['device'])  # Move loss to device
    routing_loss = torch.nn.CrossEntropyLoss().to(data['device'])  # Move loss to device

    # class_indices = defaultdict(list)
    # for i, label in enumerate(data['ds_train'].targets):
    #     class_indices[int(label)].append(i)
    #
    # clss_data = {}
    # for class_label, indices in class_indices.items():
    #     clss_data[class_label] = MyDataset(Subset(data['ds_train'], indices))

    losses = []
    accs = []

    # Initialize networks and load state dicts
    for key in range(param['num_clients']):
        backbone = SharedBackbone()
        clients[key] = (client_network_cnn(key, param['n_layers'], param['input_dim'], param['hidden_dim'], param['output_dim'], backbone, DEVICE=data['device']))
        # Load state dict
        # Using map_location to ensure the loaded model is on the current device
        clients[int(key)].load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'client_base_{int(key)}.pt'), map_location=data['device'], weights_only=False))

    with torch.no_grad():
        for key, client in clients.items():
            client_loss = []
            client_acc = []
            for x, y in data['test_loader']:
                x = x.to(data['device'])  # Move input to device
                y = y.to(data['device'])  # Move labels to device
                y_1hot = one_hot_encode(y.cpu(), len(param['clss'])).to(data['device'])  # Create one-hot on device
                encoded = clients[int(key)].encode(x)
                #encoded = torch.flatten(encoded, start_dim=1)
                o = clients[int(key)](encoded)

                # print(torch.argmax(o, dim=1), y_1hot)
                # los = loss(o[:,:-1], y_1hot)
                los = loss(o, y_1hot)
                acc = accuracy(o, y_1hot)
                client_loss.append(los)
                client_acc.append(acc)
            # Move loss to CPU for numpy sum/average calculation (optional but safer)
            print(key, sum(l.item() for l in client_loss) / len(client_loss))
            losses.append(sum(l.item() for l in client_loss) / len(client_loss))
            print(key, sum(client_acc) / len(client_acc))
            accs.append(sum(client_acc) / len(client_acc))
    print(losses)
    print(sum(losses) / len(losses))
    print(key, sum(client_acc) / len(client_acc))
    accs.append(sum(client_acc) / len(client_acc))

    with open(os.path.join(exp_dir, 'results', 'base_metrics.txt'), 'w') as f:
        f.write(str(sum(losses) / len(losses)))
        f.write(str(sum(accs) / len(accs)))

    losses = []
    accs = []
    # Re-initialize networks and load state dicts
    for i in range(param['num_clients']):
        backbone = SharedBackbone()
        clients[i] = (client_network_cnn(i, param['n_layers'], param['input_dim'], param['hidden_dim'], param['output_dim'], backbone, DEVICE=data['device']))
        routing_nets[i] = (routing_network(i, param['route_n_layers'], param['input_dim'], param['route_hidden_dim'], param['num_clients'],  backbone, DEVICE=data['device']))

    for key in range(param['num_clients']):
        # Load state dicts
        clients[int(key)].load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'client_{int(key)}.pt'), map_location=data['device'], weights_only=False))
        print(list(clients[key].cnn.parameters()))
        routing_nets[int(key)].load_state_dict(torch.load(os.path.join(exp_dir, 'models', f'routing_{int(key)}.pt'), map_location=data['device'], weights_only=False))

    with torch.no_grad():
        for key, client in clients.items():
            test_los(clients, routing_nets[int(key)], key, data['test_loader'], data['device'], param['clss'],
                     param['entropy_threshold'])
            client_loss = []
            client_acc = []
            for x, y in data['test_loader']:
                x = x.to(data['device'])  # Move input to device
                y = y.to(data['device'])  # Move labels to device
                y_1hot = one_hot_encode(y.cpu(), len(param['clss'])).to(data['device'])  # Create one-hot on device
                encoded = clients[int(key)].encode(x)
                #encoded = torch.flatten(encoded, start_dim=1)
                #x = torch.flatten(x, start_dim=1)
                o = clients[int(key)](encoded)

                o_max = torch.softmax(o, dim=1)

                log_o_max = torch.log(o_max)
                entropy_values = -torch.sum(o_max * log_o_max, dim=1)
                mask = entropy_values > param['entropy_threshold']
                print(entropy_values)
                # mask = torch.zeros(o_max.size()[0], device=data['device'])  # Create mask on device
                # for i, p in enumerate(o_max):
                #     entropy_value = entropy(p.detach().cpu().numpy())
                #     print(entropy_value)
                #     if entropy_value > param['entropy_threshold']:
                #         mask[i] += 1
                # mask = mask.bool()
                extracted = x[mask]
                encoded_extracted = encoded[mask]
                prediction = []
                if len(extracted) > 0:
                    routed = torch.argmax(routing_nets[key](extracted), dim=1)

                    for i, (sample, cli) in enumerate(zip(encoded_extracted, routed)):
                        prediction.append(clients[int(cli)](sample))
                    prediction = torch.stack(prediction)
                    routed = None
                    # print(torch.argmax(o, dim=1), y_1hot)
                    o[mask] = prediction
                for xo, yo in zip(o, y_1hot):
                    conf_mat[torch.argmax(yo)][torch.argmax(xo)] += 1

                row_sums = conf_mat.sum(axis=1)
                row_sums_reshaped = row_sums[:, np.newaxis]
                conf_mat = conf_mat / row_sums_reshaped
                with open(os.path.join(exp_dir, 'results', f'confusion_{key}.txt'), 'w') as f:
                    f.write(str(conf_mat))
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
    with open(os.path.join(exp_dir,'results','metrics.txt'), 'w') as f:
        f.write(str(sum(losses)/len(losses)))
        f.write(str(sum(accs)/len(accs)))

    print(losses)
    print(sum(losses) / len(losses))
    print(accs)
    print(sum(accs) / len(accs))

def freeze_weights(model):
    for name, p in model.named_parameters():
        p.requires_grad = False