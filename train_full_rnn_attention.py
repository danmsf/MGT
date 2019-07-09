from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np


class LstmEncoder(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions=100, output_dimension=1, nb_layers=1, batch_size=1, bidirectional=False):
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
        print(X.shape)
        print(X_lengths)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # X = self.last_timestep(X, X_lengths)

        return X, self.hidden

class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(hidden_size + output_size, 1)
        self.lstm = nn.LSTM(hidden_size + vocab_size,
                            output_size)  # if we are using embedding hidden_size should be added with embedding of vocab size
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        # self.nb_lstm_layers, self.batch_size
        return (torch.zeros(1, 1, self.output_size),
                torch.zeros(1, 1, self.output_size))

    def forward(self, decoder_hidden, encoder_outputs, input):
        weights = []
        for i in range(len(encoder_outputs)):
            print(decoder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(self.attn(torch.cat((decoder_hidden[0][0],
                                                encoder_outputs[i]), dim=1)))
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))

        input_lstm = torch.cat((attn_applied[0], input[0]),
                               dim=1)  # if we are using embedding, use embedding of input here instead

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        output = self.final(output[0])


        return output, hidden, normalized_weights


# class FnnAttention():
bidirectional = False
c = LstmEncoder(input_dimensions=10, hidden_dimensions=20, bidirectional=bidirectional, batch_size=1)
c.init_hidden()
x = torch.randn(10).unsqueeze(0).unsqueeze(2)
a, b = c.forward(x, X_lengths=[10])
print(a.shape)
print(b[0].shape)
print(b[1].shape)

x = AttentionDecoder(20 * (1 + bidirectional), 25, 30)
y, z, w = x.forward(x.init_hidden(), torch.cat((a, a)), torch.zeros(1, 1, 30))  # Assuming <SOS> to be all zeros
print(y.shape)
print(z[0].shape)
print(z[1].shape)
print(w)