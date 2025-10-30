import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *  # DEVICE is imported from utils
import neat
import pickle
import os

# import visualize

# NEAT configuration file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')


# The main function to run NEAT
def eval_genomes(genomes, config):
    print(genomes)
    for genome_id, genome in genomes:
        # Note: NEAT's network creation is not directly tied to PyTorch device management
        # It's an external library; assuming it handles its internal state or uses CPU
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        total_correct = 0
        total_samples = 0
        conf_mat = np.zeros((NUM_CLIENTS, len(CLSS)))
        # We'll evaluate the network on the test set to measure its fitness
        with torch.no_grad():
            for t, (x, y) in enumerate(test_loader):
                if t == 200:
                    break
                x = x.to(DEVICE)  # Move input to device
                y = y.to(DEVICE)  # Move labels to device
                # print(x.shape)
                x = x.view(x.size(0), -1).float()
                # print(x.shape)

                # NEAT activation requires numpy or list-like input for a single sample
                # Need to move to CPU for NEAT evaluation
                o = net.activate(x[0].cpu().numpy())
                # print(o)
                o_max = np.argmax(o)

                routed_correct = 0
                conf_mat[o_max][int(y[0].cpu())] += 1  # Move label to CPU for indexing
                if int(y[0].cpu()) in main_clss_dict[o_max]:
                    routed_correct += 1
                    # routed_output = clients[int(client_id)](x[idx])
                    # routed_prediction = torch.argmax(routed_output)
                    # if routed_prediction == y[idx]:
                    #     routed_correct += 1
                else:
                    routed_correct -= 1
                total_correct += routed_correct
                total_samples += 1

        # Fitness is a measure of routing accuracy
        genome.fitness = total_correct / total_samples if total_samples > 0 else 0
        print('==========================')
        print(conf_mat)
        print('==========================')


clients = {}
routing_nets = {}
for i in range(NUM_CLIENTS):
    # client_network and routing_network move themselves to DEVICE in their __init__
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM))
    routing_nets[i] = (routing_network(i, 10, INPUT_DIM, 25, NUM_CLIENTS))

loss = torch.nn.CrossEntropyLoss().to(DEVICE)  # Move loss function to device
routing_loss = torch.nn.CrossEntropyLoss().to(DEVICE)  # Move loss function to device
# Load the NEAT configuration
# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      config_path)
# #
# # # Create the population, which is the top-level object for the NEAT algorithm
# p = neat.Population(config)
#
# # Add a stdout reporter to show progress in the terminal
# p.add_reporter(neat.StdOutReporter(True))
# stats = neat.StatisticsReporter()
# p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(5))

class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))

main_clss_dict = defaultdict(list)

mixed_data = create_mixed_datasets(clss_data, num_of_clients=NUM_CLIENTS, num_main_classes=2, rnd_ratio=0.8,
                                   datasets_len=10000)

for id, dat in mixed_data.items():
    for cls in dat.main_clss:
        main_clss_dict[cls].append(id)
    mixed_data[id] = (DataLoader(dat, batch_size=128, shuffle=True), dat.main_clss)
if client_train:
    for key, (loader, main_clss) in mixed_data.items():
        optim = torch.optim.SGD(clients[int(key)].parameters(), lr=0.1)
        for epoch in range(20):
            print(key, epoch)
            for x, y in loader:
                x = x.to(DEVICE)  # Move input to device
                y = y.to(DEVICE)  # Move labels to device
                y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
                x = torch.flatten(x, start_dim=1)
                o = clients[int(key)](x)
                # print(torch.argmax(o, dim=1), y_1hot)
                los = loss(o, y_1hot)
                # if len(routed):
                #     los += loss(routed, y_1hot[mask])
                los.backward()
                optim.step()
                optim.zero_grad()
                # print(los)

        torch.save(clients[int(key)].state_dict(), f'client_{int(key)}.pt')  # Save only state dict
        # Re-loading with weights_only=False requires saving the whole model,
        # but saving state dict is better practice. Adjusting loading in test.py
else:
    for key, (loader, main_clss) in mixed_data.items():
        # Load state dict and then move to device, or rely on client_network init
        clients[int(key)].load_state_dict(torch.load(f'./client_{int(key)}.pt', map_location=DEVICE, weights_only=True))

for key, (loader, main_clss) in mixed_data.items():
    routing_optim = torch.optim.AdamW(routing_nets[int(key)].parameters(), lr=0.001)
    client_optim = torch.optim.AdamW(clients[int(key)].parameters(), lr=0.001)
    for epoch in range(35):
        print(key, epoch)
        for x, y in loader:
            # for name, param in clients[int(key)].named_parameters():
            #     print(name,param)
            x = x.to(DEVICE)  # Move input to device
            y = y.to(DEVICE)  # Move labels to device
            y_1hot = one_hot_encode(y.cpu(), len(CLSS)).to(DEVICE)  # Create one-hot on device
            x = torch.flatten(x, start_dim=1)
            o = clients[int(key)](x)
            routed, mask = entropy_routing(routing_nets[int(key)], clients, x, o, y, main_clss_dict, routing_loss,
                                           routing_optim)
            # print(torch.argmax(o, dim=1), y_1hot)
            los = loss(o, y_1hot)
            # if len(routed):
            #     los += loss(routed, y_1hot[mask])
            los.backward()
            client_optim.step()
            client_optim.zero_grad()
        if epoch % 10 == 0:
            test_los(clients, routing_nets[int(key)], loss, key)
    torch.save(routing_nets[int(key)].state_dict(), f'routing_{int(key)}.pt')  # Save only state dict
#
# # winner = p.run(eval_genomes, 6)
# #
# # print('\nBest genome:\n{!s}'.format(winner))
# #
# # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
# #
# # # Visualize the network and stats
# # node_names = {i: str(i) for i in range(NUM_CLIENTS)}
# # # visualize.draw_net(config, winner, True, node_names=node_names)
# # # visualize.plot_stats(stats, ylog=False, view=True)
# # # visualize.plot_species(stats, view=True)
# #
# # # Save the winner
# # with open('winner.pkl', 'wb') as f:
# #     pickle.dump(winner, f)
# #
# # eval_genomes([(1,winner)], config)
# # print(main_clss_dict)