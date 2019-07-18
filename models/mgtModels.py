from torch import nn
import torch
from torch.autograd import Variable


class FcModel(nn.Module):
    def __init__(self
              ,input_dimensions
              ,hidden_dimenssions
              ,output_dimenssions):
        super(FcModel, self).__init__()

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


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np


class FullLSTM(nn.Module):
    def __init__(self, input_dimensions, output_dimension=1, hidden_dimensions=100, nb_layers=1, batch_size=3):
        super(FullLSTM, self).__init__()

        self.nb_lstm_layers = nb_layers
        self.hidden_dimensions = hidden_dimensions
        self.input_dimensions = input_dimensions
        self.output_dimension = output_dimension
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            input_size=self.input_dimensions,
            hidden_size=self.hidden_dimensions,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
            )

        self.linear = nn.Linear(self.hidden_dimensions, self.output_dimension)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.on_gpu = False

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, hidden_dimensions)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.hidden_dimensions)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.hidden_dimensions)

        if self.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        # this is only correct for batch_frits = True ; otherwise change indexing in 47 and 49
        lengths = torch.LongTensor(lengths)
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        # X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dimensions)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.last_timestep(X, X_lengths)


        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, hidden_dimensions) -> (batch_size * seq_len, hidden_dimensions)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # X = X.contiguous()
        # X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.linear(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_dimensions) -> (batch_size, seq_len, output_dimension)
        # X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, output_dimension)
        # X = X.view(batch_size, seq_len, self.output_dimension)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.output_dimension)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        # tag_pad_token = self.tags['<PAD>']
        # mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        # nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        # Y_hat = Y_hat[range(Y_hat.shape[0])]
        x_max = np.max(X_lengths)
        mask = np.array(range(len(X_lengths)))*x_max + X_lengths -1
        # mask = [s - 1 for s in X_lengths]
        # mask

        Y_hat = Y_hat[mask, 0]


        # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        loss = self.criterion(Y_hat, Y)
        return loss, Y_hat
