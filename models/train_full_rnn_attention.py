from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import json
import os

settings = json.loads(open(os.getcwd() + "/params.json").read())
fp = settings['filepaths']


class LstmEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions=100, output_dimension=1, nb_layers=1, batch_size=1,
                 bidirectional=False):
        super(LstmEncoder, self).__init__()

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
            bidirectional=bidirectional
        )

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
        print(X.shape)
        print(X_lengths)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X_last = self.last_timestep(X, X_lengths)

        return X, X_last, self.hidden


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
