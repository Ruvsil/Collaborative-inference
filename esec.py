from config import *
from utils import test_los, test
from main import main
import os
import json
np.set_printoptions(suppress=True, precision=2)
exp_name = "prova"
exp_path = os.path.join(os.getcwd(), 'experiments')
c = 0
while os.path.exists(os.path.join(exp_path, exp_name)):
    exp_name += str(c)
    c += 1

exp_dir = os.path.join(exp_path, exp_name)
res_dir = os.path.join(exp_dir, 'results')
model_dir = os.path.join(exp_dir, 'models')
if not os.path.exists(exp_path):
    os.mkdir(exp_path)
os.mkdir(exp_dir)
os.mkdir(res_dir)
os.mkdir(model_dir)

params_dict, data_dict = get_params()
with open(os.path.join(exp_dir, 'configuration.json'), 'w') as f:
    json.dump(params_dict, f)
clients, routing_nets = main(params_dict, data_dict, exp_dir)
for key, cl in enumerate(clients):
    test_los(clients, routing_nets[int(key)], key, data_dict['test_loader'], data_dict['device'], params_dict['clss'], params_dict['entropy_threshold'])

test(exp_dir,params_dict,data_dict)