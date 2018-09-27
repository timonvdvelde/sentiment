import torch
import torch.nn as nn
import torch.nn.functional as F


class TimonNet(nn.Module):
    def __init__(self, dimensions, filters=100, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv1d(1, filters, 3*dimensions, stride=dimensions)
        self.conv2 = nn.Conv1d(1, filters, 4*dimensions, stride=dimensions)
        self.conv3 = nn.Conv1d(1, filters, 5*dimensions, stride=dimensions)

        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(3 * filters, 2)


    def forward(self, x):
        x1 = self.tanh(self.conv1(x))
        x2 = self.tanh(self.conv2(x))
        x3 = self.tanh(self.conv3(x))

        x1 = F.max_pool1d(x1, x1.size()[2])
        x2 = F.max_pool1d(x2, x2.size()[2])
        x3 = F.max_pool1d(x3, x3.size()[2])

        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)
        x3 = x3.squeeze(2)

        x = torch.cat([x1, x2, x3], 1)

        x = self.dropout(x)
        x = self.linear(x)
        return x

