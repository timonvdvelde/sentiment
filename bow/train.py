import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import model as md
import preprocess as pre

# Loading data
file_review_vectors = 'review_vectors.json'
inputs, targets = pre.load_review_vectors(file_review_vectors)

# Temporary lol
v_dim = np.shape(inputs)[1]
hidden_layers = []
classes = np.shape(targets)[1]
learning_rate = 0.01
moment = 0.9
use_cuda = False
num_epochs = 10

# training
model = md.MLPNet(n_input=v_dim,
                  n_hidden=hidden_layers,
                  n_output=classes)

if use_cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    accuracy = 0
    for enum, i in enumerate(np.random.permutation(range(np.shape(inputs)[0]))):
        x = torch.from_numpy(inputs[i]).float()
        y = torch.from_numpy(targets[i]).long().reshape(1,2)

        optimizer.zero_grad()

        prediction = model.forward(x)
        loss = criterion(prediction, y.max(1)[1])
        loss.backward()
        optimizer.step()

        if prediction.max(1)[1] == y.max(1)[1]:
            accuracy += 1

    # # test for accuracy
    # accuracy = 0
    # total = 1000
    # for i_a in np.random.randint(0, np.shape(inputs)[0], size=total):
    #     x_a = torch.from_numpy(inputs[i_a]).float()
    #     y_a = torch.from_numpy(targets[i_a]).long().reshape(1,2)
    #
    #     prediction_a = model.forward(x_a)
    #     if prediction_a.max(1)[1] == y_a.max(1)[1]:
    #         accuracy += 1
    print("the accuracy is", accuracy/np.shape(inputs)[0])


torch.save(model.state_dict(), model.name())
