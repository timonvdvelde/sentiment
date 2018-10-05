import sys

import numpy as np
import torch.utils.data as data
import preprocess as pre


class BowDataset(data.Dataset):

    def __init__(self, path):
        self.inputs, self.targets = pre.load_review_vectors(path)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
