import os
import json
import numpy as np
import sys
import shutil

# To import functions from config.py and utils.py without modifying sys.path
# Since the scripts are likely in the same directory, this should work.

# --- Import Refactoring (Assuming you can make a small change to config.py) ---
# NOTE: The provided config.py uses get_params to define the configuration.
# To run multiple experiments easily, we'll redefine the core logic of
# get_params here to handle different configurations directly.
from config import get_params as default_get_params
from utils import test_los, test
from main import main
import torch

np.set_printoptions(suppress=True, precision=2)


def run_single_experiment(params_override, base_exp_name="prova"):
    """
    Runs a single experiment based on the provided parameters override.
    This function mimics the logic of esec.py but uses a passed-in config.
    """

    # 1. Get base parameters and override with experiment-specific ones
    # Use the default get_params to get the base data dictionary
    # and then merge/override the parameters.
    default_params, data_dict = default_get_params()
    params_dict = default_params.copy()
    params_dict.update(params_override)

    # Re-calculate the device and test_loader in case the dataset changed
    # (Although in this example, only 'mnist' is defined in get_params)
    data_path = './datasets'
    if params_dict['dataset'] == 'mnist':
        from torchvision.datasets import MNIST
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        ds_test = MNIST(data_path, train=False, download=True, transform=transform)
    elif params_dict['dataset'] == 'cifar':
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        ds_test = CIFAR10(data_path, train=False, download=True, transform=transform)

    data_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict['test_loader'] = torch.utils.data.DataLoader(ds_test, batch_size=2048, shuffle=True)

    # 2. Setup Experiment Directory
    exp_path = os.path.join(os.getcwd(), 'experiments')
    exp_name = base_exp_name
    c = 0
    # Ensure the experiment directory name is unique
    while os.path.exists(os.path.join(exp_path, exp_name + str(c))):
        c += 1

    final_exp_name = exp_name + str(c)
    exp_dir = os.path.join(exp_path, final_exp_name)
    res_dir = os.path.join(exp_dir, 'results')
    model_dir = os.path.join(exp_dir, 'models')

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    os.mkdir(exp_dir)
    os.mkdir(res_dir)
    os.mkdir(model_dir)

    print(f"\n--- Starting Experiment: {final_exp_name} ---")

    # 3. Save Configuration
    with open(os.path.join(exp_dir, 'configuration.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)

    # 4. Run Main Training Logic
    clients, routing_nets = main(params_dict, data_dict, exp_dir)

    # 5. Run Per-Client Testing (from esec.py)
    # for key, cl in enumerate(clients):
    #     # The key is an index, use params_dict['clss'] since it's defined
    #     test_los(clients, routing_nets[int(key)], key, data_dict['test_loader'],
    #              data_dict['device'], params_dict['clss'], params_dict['entropy_threshold'])

    # 6. Run Final Comprehensive Testing (from esec.py)
    test(exp_dir, params_dict, data_dict)

    print(f"--- Experiment {final_exp_name} Finished ---")
    print(f"Results saved to: {exp_dir}")
    print("-" * 50)


if __name__ == '__main__':

    # Define your multiple configurations here.
    # Each dictionary only needs to specify the parameters that change
    # from the default values in config.py.

    configurations = [
        {
            # Configuration 1: Baseline (Matches defaults closely)
            "base_exp_name": "Baseline_Default_Config",
            "entropy_threshold": 0.3,
            "num_clients": 10,
            "num_main_clss": 3,
            "main_clss_percent": 0.8,
            "n_layers": 4,
            "route_n_layers": 10,
            "hidden_dim": 2048,  # Default value
            "route_hidden_dim": 2048  # Default value
        },
        {
            # Configuration 2: High Routing Sensitivity (Low Entropy Threshold)
            # Low threshold means more samples are considered "uncertain" and routed.
            "base_exp_name": "High_Routing_Sensitivity_Entropy_0_1",
            "entropy_threshold": 0.1,  # Lowered from 0.3
            "num_clients": 10,
            "num_main_clss": 3,
            "main_clss_percent": 0.8,
            "n_layers": 4,
            "route_n_layers": 10,
            "hidden_dim": 2048,
            "route_hidden_dim": 2048
        },
        {
            # Configuration 3: High Client Specialization (Fewer Main Classes)
            # Clients are more specialized, should rely more on routing for unseen classes.
            "base_exp_name": "High_Specialization_MainClss_2",
            "entropy_threshold": 0.3,
            "num_clients": 10,
            "num_main_clss": 2,  # Lowered from 3
            "main_clss_percent": 0.8,
            "n_layers": 4,
            "route_n_layers": 10,
            "hidden_dim": 2048,
            "route_hidden_dim": 2048
        },
        {
            # Configuration 4: Deep Client Network, Shallow Routing Network, Smaller Hidden Dim
            # Test if a more complex client network with fewer neurons and a simpler router is sufficient.
            "base_exp_name": "Deep_Client_Shallow_Router_Small_Dim",
            "entropy_threshold": 0.3,
            "num_clients": 10,
            "num_main_clss": 3,
            "main_clss_percent": 0.8,
            "n_layers": 6,  # Increased from 4
            "route_n_layers": 5,  # Decreased from 10
            "hidden_dim": 1024,  # Varied from 2048
            "route_hidden_dim": 1024  # Varied from 2048
        },
        {
            # Configuration 5: Low Non-IID Data Distribution (Higher Main Class Percentage)
            # Training data is less diverse, potentially leading to higher entropy on test data.
            "base_exp_name": "High_MainClss_Percent_0_95",
            "entropy_threshold": 0.3,
            "num_clients": 10,
            "num_main_clss": 3,
            "main_clss_percent": 0.95,  # Increased from 0.8 (95% of data is main classes)
            "n_layers": 4,
            "route_n_layers": 10,
            "hidden_dim": 2048,
            "route_hidden_dim": 2048
        }
    ]

    for i, config_params in enumerate(configurations):
        # Extract the specific experiment name for unique directory creation
        exp_name = config_params.pop("base_exp_name")
        print(f"Configuration {i + 1}/{len(configurations)}: Running {exp_name}")

        try:
            run_single_experiment(config_params, base_exp_name=exp_name)
        except Exception as e:
            print(f"An error occurred during experiment '{exp_name}': {e}")
            # Optionally, you can decide whether to stop or continue to the next experiment
            # sys.exit(1)
            continue  # Continue to the next configuration on failure