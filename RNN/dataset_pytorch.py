import torch.nn as nn
import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    en_model = {}
    def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = np.array([Dataset.en_model[self.data[index][i]] for i in range(len(self.data[index])) if self.data[index][i] in Dataset.en_model])
        y = self.labels[index]
        return X, y