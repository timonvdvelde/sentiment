#!/usr/bin/env python3

from timonNet import TimonNet
from dataset import ReviewDataset
import preprocess
import os
import torch
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.utils.data import DataLoader, SubsetRandomSampler
import random

path_data = '../data/train/'
path_pos = path_data + 'pos/'
path_neg = path_data + 'neg/'

path_embed = '../embeddings/'
file_embed_raw = 'glove.twitter.27B.25d.txt'
file_embed_json = 'glove.twitter.27B.25d.json'

dimensions = 25
batch_size = 50
params_file = 'params'


def get_dataset():
    embeddings = preprocess.load_embed(path_embed + file_embed_json)
    files = []
    labels = []

    for sentiment in (path_pos, path_neg):
        for dirpath, dirnames, filenames in os.walk(sentiment):
            for filename in filenames:
                files.append(dirpath + '/' + filename)
                
                if sentiment == path_neg:
                    labels.append(0)
                elif sentiment == path_pos:
                    labels.append(1)
    
    data = ReviewDataset(embeddings, files, labels)
    return data


def validate(validation_loader, net):
    net.eval()
    avg_loss = 0
    avg_accuracy = 0
    
    for i, (vectors, targets) in enumerate(validation_loader):
        logits = net(vectors)
        loss = F.cross_entropy(logits, targets)
        corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).sum()
        accuracy = 100.0 * corrects / len(vectors)
        
        avg_loss += loss.data
        avg_accuracy += accuracy.data
    
    avg_loss /= i
    avg_accuracy /= i
    
    print('Validation')
    print('Loss:', avg_loss)
    print('Accuracy:', avg_accuracy)
    

def train(train_loader, validation_loader, net):
    optimizer = Adadelta(net.parameters())
    validate(validation_loader, net)
    
    for i in range(50):
        print('Epoch:', i)
        net.train()

        for i, (vectors, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = net(vectors)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        validate(validation_loader, net)
        torch.save(net.state_dict(), params_file)


def collate(items):
    vectors = [item[0] for item in items]
    targets = [item[1] for item in items]
    maxlength = max([len(vector) for vector in vectors])

    for i in range(len(vectors)):
        padding = maxlength - len(vectors[i])
        vectors[i] = torch.cat((vectors[i], torch.zeros(padding)))
  
    batch_data = torch.zeros((len(vectors), 1, maxlength))
    batch_targ = torch.zeros(len(vectors), dtype=torch.long)

    for i in range(len(vectors)):
        batch_data[i][0] = vectors[i]
        batch_targ[i] = targets[i]

    return batch_data, batch_targ


def main():
    print("Getting all the data.")
    data = get_dataset()

    indices = list(range(len(data)))
    random.shuffle(indices)
    train_size = int(0.9 * len(data))
    train_indices = indices[:train_size]
    train_sampler = SubsetRandomSampler(train_indices)

    validation_indices = indices[train_size:]
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    train_loader = DataLoader(data,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=collate)
    validation_loader = DataLoader(data,
                                   batch_size=batch_size,
                                   sampler=validation_sampler,
                                   collate_fn=collate)
    print("Data got.")

    net = TimonNet(dimensions)
    train(train_loader, validation_loader, net)


if __name__ == '__main__':
    main()

