from config import *
from main import main
import os
import json

exp_name = "name"
exp_path = os.path.join(os.getcwd(),'experiments')
c = 0
while os.path.exists(os.path.join(exp_path, exp_name)):
    exp_name += str(c)
    c += 1

exp_dir = os.path.join(exp_path,exp_name)
res_dir = os.path.join(exp_dir, 'results')
model_dir = os.path.join(exp_dir, 'models')
if not os.path.exists(exp_path):
    os.mkdir(exp_path)
os.mkdir(exp_dir)
os.mkdir(res_dir)
os.mkdir(model_dir)

params_dict = get_params()
clients, routing_nets = main(params_dict, exp_dir)
with open(os.path.join(exp_dir,'configuration.json')) as f:
    json.dump(params_dict,f)
