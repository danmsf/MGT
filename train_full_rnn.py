from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
from mgtUtils import plot_loss

class FullLSTM(nn.Module):
    def __init__(self, input_dimensions, output_dimension = 1, hidden_dimensions=100, nb_layers=1, batch_size=3):
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

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, hidden_dimensions) -> (batch_size * seq_len, hidden_dimensions)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.linear(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_dimensions) -> (batch_size, seq_len, output_dimension)
        # X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, output_dimension)
        X = X.view(batch_size, seq_len, self.output_dimension)

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
        mask = [s - 1 for s in X_lengths]
        Y_hat = Y_hat[mask, 0]


        # compute cross entropy loss which ignores all <PAD> tokens
        # ce_loss = -torch.sum(Y_hat) / nb_tokens
        loss = self.criterion(Y_hat, Y)
        return loss


seqs = [[1,2,3],[4,5,6,7,8,9], [4,3,2,1]]
seqs_len = [len(s) for s in seqs]
max_len = max(seqs_len)
seqs_pad = [s + [0]*(max_len - len(s)) for s in seqs]
target = np.random.normal(size=len(seqs))

# (batch_size, seq_len, embedding_dim)
x = torch.Tensor(seqs_pad)
x = x.unsqueeze(2)

lr = 1e-04
decay = 0.9


model = FullLSTM(input_dimensions=1, batch_size=3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

losses = []
for i in range(1000):
    yhat = model(x, seqs_len)
    loss = model.loss(yhat, torch.Tensor(target), seqs_len)
    print(i)
    losses.append(loss)
    optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()


plot_loss(losses)