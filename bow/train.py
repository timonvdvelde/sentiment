import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import model as md
import preprocess as pre
from dataset import BowDataset

# Change parameters here
embedding_size = 25
hidden_layers = [100]
classes = 2
num_epochs = 10
embedding_type = "twitter"

def evaluate(data_loader, net):
    """
    Prints loss and accuracy for a given dataset.
    """
    net.eval()

    accuracy = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(data_loader):
        x = x.float()
        y = y.long()

        prediction = net.forward(x)
        loss += criterion(prediction, y.max(1)[0])

        accuracy += sum(prediction.max(1)[1] == y.max(1)[0]).float()/len(x)

    loss /= i+1
    accuracy /= i+1

    print('Loss:', loss)
    print('Accuracy:', accuracy)
    return accuracy

def test(net=None):
    """
    Evaluates network on test data.
    """
    file_review_vectors = 'test_vectors_{}_{}.json'.format(embedding_type,
                                                           embedding_size)
    test_data = BowDataset(file_review_vectors)
    test_loader = DataLoader(test_data,
                              batch_size=64,
                              shuffle=True)

    if not net:
        net = md.MLPNet(embedding_size, hidden_layers, classes)
        net.load_state_dict(torch.load("{}_{}_{}_{}".format(embedding_type,
                                                            embedding_size,
                                                            hidden_layers,
                                                            classes)))
        evaluate(test_loader, net)
    else:
        evaluate(test_loader, net)

def train(train_loader, v_dim, hidden_layers, classes,
          learning_rate, moment, use_cuda, num_epochs):
    model = md.MLPNet(n_input=v_dim,
                      n_hidden=hidden_layers,
                      n_output=classes,
                      name="{}_{}_{}_{}".format(embedding_type,
                                                embedding_size,
                                                hidden_layers,
                                                classes))

    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        accuracy = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float()
            y = y.long()

            optimizer.zero_grad()

            prediction = model.forward(x)
            loss = criterion(prediction, y.max(1)[0])
            loss.backward()
            optimizer.step()

            accuracy += sum(prediction.max(1)[1] == y.max(1)[0]).long()

        print("the accuracy is", loss, accuracy.float()/25000.0)
    torch.save(model.state_dict(), model.name())


def main():

    # Loading data
    file_review_vectors = 'train_vectors_{}_{}.json'.format(embedding_type,
                                                            embedding_size)
    train_data = BowDataset(file_review_vectors)
    train_loader = DataLoader(train_data,
                              batch_size=64,
                              shuffle=True)

    train(train_loader,
          v_dim = embedding_size,
          hidden_layers = hidden_layers,
          classes = classes,
          learning_rate = 0.01,
          moment = 0.9,
          use_cuda = False,
          num_epochs = num_epochs)

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'test':
        test()
    else:
        print("Wat duh?")
