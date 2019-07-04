from torch import nn
import torch
from torch.autograd import Variable


class FcModel(nn.Module):
    def __init__(self
              ,input_dimensions
              ,hidden_dimenssions
              ,output_dimenssions):
        super(Model, self).__init__()

        self.input = nn.Linear(input_dimensions, hidden_dimenssions)
        self.hidden = nn.Linear(hidden_dimenssions, hidden_dimenssions)
        self.output = nn.Linear(hidden_dimenssions, output_dimenssions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.sigmoid(self.input(x))
        h2 = self.sigmoid(self.hidden(h1))
        y_pred = self.output(h2)
        return y_pred


class RnnModel(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, use_cuda):
        super(RnnModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # self.enc_init_state = self.init_hidden(hidden_dim)
        self.batch_size = batch_size
        self.num_layers = 1

    def forward(self, inputs):
        # input : [time * batch_size * input_dimenssion]
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        # y_hat = self.linear(lstm_out)
        y_pred = self.linear(lstm_out[-1])
        return y_pred[:, 0]

    def init_hidden(self):
        # """Trainable initial hidden state"""
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
         torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

