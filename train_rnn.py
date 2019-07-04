import argparse
import pprint as pp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from dataset import  DatasetLstmMGT
from torch.utils.data import DataLoader
import os
from mgtModels import RnnModel
from mgtUtils import plot_loss, plot_pred, split_sample


def str2bool(v):
      return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description="Recurrent Neural Network")

# Data
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--lr', default=1e-5, help='')
parser.add_argument('--decay', default=0.9, help='')
parser.add_argument('--hidden_dimensions', default=800, help='')
parser.add_argument('--epochs', default=100, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--random_seed', default=1234, help='')
parser.add_argument('--use_cuda', type=str2bool, default=False, help='')
parser.add_argument('--model_path', type=str, default='C:\\Users\\Dan\\PycharmProjects\\MGT\\saved_models\\rnn')
parser.add_argument('--data_path', type=str, default='C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\panel_stages.csv')

args = vars(parser.parse_args())

# Pretty print the run args
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))


# if not args['disable_tensorboard']:
#     if args['active_search']:
#         configure(os.path.join(args['log_dir'], args['task'],'active_search', args['run_name']))
#     else:
#         configure(os.path.join(args['log_dir'], args['task'], args['run_name']))


batch_size = args['batch_size']
lr = args['lr']
decay = args['decay']
hidden_dimensions = args['hidden_dimensions']
epochs = args['epochs']
log_interval = args['log_interval']
model_path = args['model_path']
data_path = args['data_path']

dataset = DatasetLstmMGT(data_path)
train_loader, validation_loader = split_sample(dataset, batch_size=batch_size)

X, y = dataset.__getitem__(0)
input_dimensions = X.shape[0]
output_dimensions = 1

model_path = os.path.join(model_path, str(lr), str(hidden_dimensions))

try:
    os.makedirs(model_path)
except:
    pass

model = RnnModel(input_dimensions, hidden_dimensions, output_dimensions, batch_size=batch_size, use_cuda=False)

criterion = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)


losses = []
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs, target = Variable(data), Variable(target)
        y_pred = model(inputs.transpose(1, 2).transpose(0, 1))
        loss = criterion(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
            losses.append(loss.data.item())
    if (epoch+1) % 100 == 0:
        print(f'Saving model epoch {epoch}')
        torch.save(model.state_dict(), os.path.join(model_path,f'epoch-{epoch+1}.pt'))


# plot losses and predictions
plot_loss(losses)

predictions = []
targets = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(validation_loader):
        # data[torch.isnan(data)] = 0
        inputs, target = Variable(data), Variable(target)
        y_pred = model(inputs.transpose(1, 2).transpose(0, 1))
        predictions.extend(y_pred.numpy())
        targets.extend(target.numpy())
        loss = criterion(y_pred, target)
        print(loss)

plot_pred(predictions, targets)
