import torch
import torch.nn as nn
import copy

class MLPNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, name="test"):
        """
        A pytorch nn model
        """
        self.nam = name

        super(MLPNet, self).__init__()
        self.n_layers = copy.copy(n_hidden)
        self.n_layers.insert(0, n_input)
        self.n_layers.append(n_output)

        self.layers = nn.ModuleList()

        for i in range(len(self.n_layers) - 1):
            self.layers.append(nn.Linear(self.n_layers[i], self.n_layers[i+1]))

    def forward(self, x):

        x = x.view(-1, self.n_layers[0])
        for i in range(len(self.n_layers) - 2):
            x = nn.functional.relu(self.layers[i](x))

        # x = nn.Softmax(self.layers[-1](x))
        # x = nn.functional.softmax(self.layers[-1](x), dim=1)
        x = self.layers[-1](x)
        return x

    def name(self):
        return self.nam
