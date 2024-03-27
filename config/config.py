import yaml
import yaml
import torch
import torch.nn as nn
import numpy as np

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0])]
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def test_yaml_config_loading():
    config = load_config('config/config.yaml')  
    net_config = config['network']
    network = SimpleNN(net_config['input_size'], net_config['hidden_layers'], 
                       net_config['output_size'], net_config['activation'])
    print(network)  



if __name__ == '__main__':
    test_yaml_config_loading()
