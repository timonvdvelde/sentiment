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
import sys

#path_data = '../data/train/'
path_data = '../data/tokenized/train/'
path_pos = path_data + 'pos/'
path_neg = path_data + 'neg/'

#path_test = '../data/test/'
path_test = '../data/tokenized/test/'
path_test_pos = path_test + 'pos/'
path_test_neg = path_test + 'neg/'

path_embed = '../embeddings/'
file_embed_raw = 'glove.twitter.27B.25d.txt'
file_embed_json = 'glove.twitter.27B.25d.json'

#file_embed_raw = 'glove_vectors_unsup_movies_25d_lowercase_preservelines.txt'
#file_embed_json = 'glove_vectors_unsup_movies_25d_lowercase_preservelines.json'

dimensions = 25
batch_size = 50
params_file = 'params'


def get_dataset(embeddings, paths, val=False):
    """
    Gathers all the review files pathnames, and returns a ReviewDataset object.
    If val == True, splits the data into 90% training data and 10% validation data.
    """
    files = []
    labels = []

    for sentiment in paths:
        files.append([])
        labels.append([])

        for dirpath, dirnames, filenames in os.walk(sentiment):
            for filename in filenames:
                files[-1].append(dirpath + '/' + filename)
                
                if sentiment == path_neg or path_test_neg:
                    labels[-1].append(0)
                elif sentiment == path_pos or path_test_pos:
                    labels[-1].append(1)
   
    if val:
        split = int(len(files[0]) * 0.9)
        train_files = files[0][:split]
        train_files.extend(files[1][:split])
        train_labels = labels[0][:split]
        train_labels.extend(labels[1][:split])
        train_data = ReviewDataset(embeddings, train_files, train_labels)

        val_files = files[0][split:]
        val_files.extend(files[1][split:])
        val_labels = labels[0][split:]
        val_labels.extend(labels[1][split:])
        val_data = ReviewDataset(embeddings, val_files, val_labels)

        return train_data, val_data

    files = [file for files_ in files for file in files_]
    labels = [label for labels_ in labels for label in labels_]

    data = ReviewDataset(embeddings, files, labels)
    return data


def evaluate(data_loader, net):
    """
    Prints loss and accuracy for a given dataset.
    """
    net.eval()
    avg_loss = 0
    avg_accuracy = 0
    
    for i, (vectors, targets) in enumerate(data_loader):
        logits = net(vectors)
        loss = F.cross_entropy(logits, targets)
        corrects = (torch.max(logits, 1)[1].view(targets.size()).data == targets.data).sum()
        accuracy = 100.0 * corrects / len(vectors)
        
        avg_loss += loss.data
        avg_accuracy += accuracy.data
    
    avg_loss /= i
    avg_accuracy /= i
    
    print('Loss:', avg_loss)
    print('Accuracy:', avg_accuracy)
    

def train(train_loader, validation_loader, net, embeddings):
    """
    Trains the network with a given training data loader and validation data
    loader.
    """
    optimizer = Adadelta(net.parameters())
    evaluate(validation_loader, net)
    
    for i in range(50):
        print('Epoch:', i)
        net.train()

        for i, (vectors, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = net(vectors)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        print('Validation')
        evaluate(validation_loader, net)
        torch.save(net.state_dict(), params_file)

    if i % 10 == 0:
        print("Testing")
        test(embeddings, net)


def collate(items):
    """
    Function for batching items. Takes care of padding where the reviews are of
    unequal length.
    """
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


def test(embeddings, net=None):
    """
    Evaluates network on test data.
    """
    print("Loading data.")
    data = get_dataset(embeddings, (path_test_pos, path_test_neg), val=False)
    test_loader = DataLoader(data,
                             batch_size=batch_size,
                             collate_fn=collate)

    if not net:
        net = TimonNet(dimensions)
        net.load_state_dict(torch.load(params_file))
        evaluate(test_loader, net)
    else:
        evaluate(test_loader, net)


def main():
    print("Getting all the data.")
    embeddings = preprocess.load_embed(path_embed + file_embed_json)
    train_data, val_data = get_dataset(embeddings, (path_pos, path_neg), val=True)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate)

    #indices = list(range(len(data)))
    #random.shuffle(indices)
    #train_size = int(0.9 * len(data))
    #train_indices = indices[:train_size]
    #train_sampler = SubsetRandomSampler(train_indices)
    #train_loader = DataLoader(data,
    #                          batch_size=batch_size,
    #                          sampler=train_sampler,
    #                          collate_fn=collate)

    #validation_indices = indices[train_size:]
    #validation_sampler = SubsetRandomSampler(validation_indices)
    #validation_loader = DataLoader(data,
    #                               batch_size=batch_size,
    #                               sampler=validation_sampler,
    #                               collate_fn=collate)
    print("Data got.")

    net = TimonNet(dimensions)
    train(train_loader, val_loader, net, embeddings)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'test':
        embeddings = preprocess.load_embed(path_embed + file_embed_json)
        test(embeddings)
    else:
        print("Wat do?")

