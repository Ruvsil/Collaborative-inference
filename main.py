import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *
import neat
import os
#import visualize

# NEAT configuration file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')


# The main function to run NEAT
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        total_correct = 0
        total_samples = 0

        # We'll evaluate the network on the test set to measure its fitness
        with torch.no_grad():
            for x, y in test_loader:
                print(x)
                x = x.view(x.size(0), -1).float()

                # Use the evolved NEAT network for routing on samples not confidently classified
                main_clients_predictions = []
                main_clients_predictions = clients[0](x)

                #main_clients_predictions = torch.stack(main_clients_predictions)
                #print(main_clients_predictions.size())
                #main_clients_predictions = main_clients_predictions.permute(1, 0, 2)
                print(main_clients_predictions)
                # Check for unclassified samples (those with a prediction == len(CLSS))
                predictions = torch.argmax(main_clients_predictions, dim=1)
                print(predictions)

                mask = predictions == len(CLSS)
                print(mask)
                if mask.sum() > 0:
                    x_extracted = x[mask]
                    y_extracted = y[mask]

                    # Feed the extracted data to the NEAT network
                    o = net.activate(x_extracted.cpu().detach().numpy().flatten())
                    o = np.array(o).reshape(len(x_extracted), NUM_CLIENTS)
                    o_max = np.argmax(o, axis=1)

                    routed_correct = 0
                    for idx, client_id in enumerate(o_max):
                        routed_output = clients[int(client_id)](x_extracted[idx])
                        routed_prediction = torch.argmax(routed_output)
                        if routed_prediction == y_extracted[idx]:
                            routed_correct += 1

                    total_correct += routed_correct
                    total_samples += len(x_extracted)

        # Fitness is a measure of routing accuracy
        genome.fitness = total_correct / total_samples if total_samples > 0 else 0


clients = {}

for i in range(NUM_CLIENTS):
    clients[i] = (client_network(i, N_LAYERS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM + 1))

loss = torch.nn.CrossEntropyLoss()

# Load the NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for the NEAT algorithm
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

class_indices = defaultdict(list)
for i, label in enumerate(ds_train.targets):
    class_indices[int(label)].append(i)

clss_data = {}
for class_label, indices in class_indices.items():
    clss_data[class_label] = MyDataset(Subset(ds_train, indices))

main_clss_dict = defaultdict(list)

mixed_data = create_mixed_datasets(clss_data, num_of_clients=NUM_CLIENTS, num_main_classes=2, rnd_ratio=0.7,
                                   datasets_len=10000)

for id, dat in mixed_data.items():
    for cls in dat.main_clss:
        main_clss_dict[cls].append(id)
    mixed_data[id] = (DataLoader(dat, batch_size=128, shuffle=True), dat.main_clss)

for key, (loader, main_clss) in mixed_data.items():
    optim = torch.optim.SGD(clients[int(key)].parameters(), lr=0.2)
    for epoch in range(5):
        for x, y in loader:
            x_reshaped = x.view(x.size(0), -1).float()
            optim.zero_grad()
            output = clients[int(key)](x_reshaped)
            l = loss(output, y)
            print(l)
            l.backward()
            optim.step()

# Run for up to 300 generations
winner = p.run(eval_genomes, 100)

print('\nBest genome:\n{!s}'.format(winner))

# You can then use the best genome to create a network for the final routing logic
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

# Visualize the network and stats
node_names = {i: str(i) for i in range(NUM_CLIENTS)}
# visualize.draw_net(config, winner, True, node_names=node_names)
# visualize.plot_stats(stats, ylog=False, view=True)
# visualize.plot_species(stats, view=True)

# Save the winner
with open('winner.pkl', 'wb') as f:
    pickle.dump(winner, f)