import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torch.nn.functional as F
import torch.optim as optim
import model as md

# Loading data

# Temporary lol
v_dim = 25
hidden_layers = [100]
classes = 2
learning_rate = 0.01
moment = 0.9
use_cuda = False

## training
model = md.MLPNet(n_input=v_dim,
                  n_hidden=hidden_layers,
                  n_output=classes)

if use_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=moment)

criterion = nn.CrossEntropyLoss()

torch.save(model.state_dict(), model.name())
