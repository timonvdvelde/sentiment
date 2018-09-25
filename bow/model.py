import torch
import torch.nn as nn
import copy

class MLPNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        """
        A pytorch nn model
        """
        super(MLPNet, self).__init__()
        self.n_layers = copy.copy(n_hidden)
        self.n_layers.insert(0, n_input)
        self.n_layers.append(n_output)

        print(self.n_layers)
        self.layers = []

        for i in range(len(self.n_layers) - 1):
            self.layers.append(nn.Linear(self.n_layers[i], self.n_layers[i+1]))


    def forward(self, x):

        x = x.view(-1, self.n_layers[0])
        for i in range(len(self.n_layers) - 2):
            x = nn.relu(self.layers[i](x))

        x = nn.Softmax(self.layers[-1](x))
        return x
