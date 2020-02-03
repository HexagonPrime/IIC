import pickle
import torch
import code.archs as archs

config_in = open("config.pickle", "rb")
config = pickle.load(config_in)
net = archs.__dict__[config.arch](config)
net.load_state_dict(torch.load("best_net.pytorch"))
net.eval()
