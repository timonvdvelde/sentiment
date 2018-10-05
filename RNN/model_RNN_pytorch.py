import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, number_of_layers=1, dropout=0.2):
        super(GRU, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_of_layers = number_of_layers

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=number_of_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        # No need for sigmoid, when using BCELOSS with logit loss this is done by itself.

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(self.number_of_layers, self.batch_size, self.hidden_size).to(self.device)
        # return torch.zeros(self.output_size, self.batch_size, self.hidden_size).to(self.device) for i in range(self.number_of_layers)

    def forward(self, x, seq_lengths):
        x = pack_padded_sequence(x, seq_lengths, batch_first=True)
        x, self.hidden = self.gru(x, self.hidden)

        output = self.hidden[-1].view(self.batch_size, self.hidden_size)
        output = self.out(output)

        return output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, number_of_layers=1, dropout=0.2):
        super(LSTM, self).__init__()
        # Information
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.number_of_layers = number_of_layers

        # Layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, num_layers=number_of_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

        # Get Hidden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # tuple([ for i in range(number_of_layers)])
        h_0 = tuple([torch.zeros(self.number_of_layers, self.batch_size, self.hidden_size).to(self.device)])
        c_0 = tuple([torch.zeros(self.number_of_layers, self.batch_size, self.hidden_size).to(self.device)])
        return h_0 + c_0

    # source:
    #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    def forward(self, x, seq_lengths):

        # now run through LSTM
        x = pack_padded_sequence(x, seq_lengths, batch_first=True)
        packed_ouput, (ht, ct) = self.lstm(x, self.hidden)
        output = self.out(ht[-1].view(self.batch_size, self.hidden_size))

        return output
