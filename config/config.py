import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def load_config(path):
    with open(path, 'r') as file:
        args = yaml.safe_load(file)
    # ==> Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
     # ==> Logger
    set_logger(result_dir)
    logging.info(args)
    if  args['wandb']==True:
        # wandb init
        pass
    return args
    


def test_yaml_config_loading():
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

    config = load_config('config/config.yaml')  
    net_config = config['network']
    network = SimpleNN(net_config['input_size'], net_config['hidden_layers'], 
                       net_config['output_size'], net_config['activation'])
    print(network)  





def create_optimizer(network, optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        return optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        return optim.SGD(network.parameters(), lr=learning_rate)

    else:
        raise ValueError("Unsupported optimizer")

def create_loss_function(loss_function_name):
    if loss_function_name == 'MSELoss':
        return nn.MSELoss()

    else:
        raise ValueError("Unsupported loss function")

def build_training_pipeline(config_path, data_loader):
    config = load_config(config_path)
    train_config = config['training']
    optimizer = create_optimizer(network, train_config['optimizer'], train_config['learning_rate'])
    loss_function = create_loss_function(train_config['loss_function'])
    epochs = train_config['epochs']



if __name__ == '__main__':
    test_yaml_config_loading()

