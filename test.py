import numpy as np
import torch
import json
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from utils import *  # DEVICE is imported from utils
from config import *
import pickle
import os

experiment = 'name012'
exp_dir = os.path.join(os.getcwd(), 'experiments', 'experiment')

with open(os.path.join(exp_dir, 'configuration.json'), 'r') as f:
    config = json.load(f)
