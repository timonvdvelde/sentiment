import read_data
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils import data
from gensim.models import KeyedVectors
from model_RNN_pytorch import GRU, LSTM

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and fill with PAD, and get label
        # padding = torch.zeros(self.max_length, self.list_IDs[index].shape[1])
        # padding[0:self.list_IDs[index].shape[0],:] = self.list_IDs[index]
        X = np.array([en_model[self.data[index][i]] for i in range(len(self.data[index])) if self.data[index][i] in en_model])
        y = self.labels[index]

        return torch.from_numpy(X).float(), torch.Tensor([y])

def get_dataloaders(train_data, train_labels, test_data, test_labels, val_size=500, batch_size=32):
    # Get Dataset and use efficient dataloaders.
    params_train = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    params_validation = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    params_test = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    # Create a validation set
    idx = np.random.randint(0, len(train_data), val_size)
    val_data, val_labels = train_data[idx], train_labels[idx]
    train_data, train_labels = np.delete(train_data, idx), np.delete(train_labels, idx)

    training_set = Dataset(train_data, train_labels)
    validation_set = Dataset(val_data, val_labels)
    test_set = Dataset(test_data, test_labels)

    training_generator = data.DataLoader(training_set, **params_train)
    validation_generator = data.DataLoader(validation_set, **params_validation)
    test_generator = data.DataLoader(test_set, **params_test)

    return training_generator, validation_generator, test_generator

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(target, prediction):
    correct = np.sum(target == prediction)
    return correct / target.shape[0]

def calculate_data_accuracy(data_generator):
    data_acc = 0
    data_counter = 0
    with torch.no_grad():
        for sample in data_generator:
            data_X, data_y = sample
            data_X, data_y = data_X.to(device), data_y.to(device)
            data_out = model.forward(data_X).view(1,-1)
            data_acc += accuracy(np.round(sigmoid(data_out.cpu().detach().numpy())), data_y.cpu().detach().numpy())
            data_counter += 1

    data_accuracy = data_acc / data_counter
    return data_accuracy



# Default parameters
batch_size = 1
DIMENSION_SIZE = 300
hidden_nodes = 200
OUTPUT_NODES = 1
max_epochs = 10
eval_freq = 50
validation_set_size = 1000
max_iter = 5000
reached_max_iter = False
# Load word embeddings
print('Loading word embeddings. This might take a while:')
en_model = KeyedVectors.load_word2vec_format('./word_embeddings/pretrained/wiki-news-300d-1M.vec')
print('Done')

# Load training data and create Datasets
train_data, train_labels, test_data ,test_labels = read_data.create_train_test_data(store_dataframe=True, pretrained=False)
training_generator, validation_generator, test_generator = get_dataloaders(train_data, train_labels, test_data, test_labels,
                                                                            val_size=validation_set_size, batch_size=batch_size)

# initialize model, loss and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GRU(DIMENSION_SIZE, hidden_nodes, OUTPUT_NODES, batch_size)
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


# training procedure
iterations = 0
train_loss = []
train_acc = 0
train_counter = 0

print('Starting training')
for epoch in range(max_epochs):
    for sample in training_generator:
         # get the inputs
        train_X, train_y = sample
        train_X, train_y = train_X.to(device), train_y.to(device)

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        train_out = model.forward(train_X).view(1,-1)
        train_out, train_y = train_out.view(train_out.numel()), train_y.view(train_y.numel())

        model.zero_grad()
        loss = loss_function(train_out,train_y)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        iterations += 1

        train_acc += accuracy(np.round(sigmoid(train_out.cpu().detach().numpy())), train_y.cpu().detach().numpy())
        train_counter += 1

        if iterations % eval_freq == 0:
            val_acc = calculate_data_accuracy(validation_generator)
            # Not very nice to calculate train acc and loss like this, maybe fix it later.
            print("Train accuracy: %.2f, Validation_accuracy: %.2f, Train_loss: %.3f " % ((train_acc / train_counter), val_acc, np.mean(np.asarray(train_loss))))
            train_acc = 0
            train_counter = 0

        if iterations % max_iter == 0:
            print('Reached maximum number of iterations')
            reached_max_iter = True
            break

    if reached_max_iter == True:
        break

test_acc = calculate_data_accuracy(test_generator)
print('Accuracy on the test set: %.2f' % test_acc)
