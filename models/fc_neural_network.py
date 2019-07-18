import argparse
import pprint as pp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DatasetMGT
from torch.autograd import Variable
import os
from mgtModels import FcModel
from mgtUtils import *


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
parser.add_argument('--model_path', type=str, default='C:\\Users\\Dan\\PycharmProjects\\MGT\\saved_models')
parser.add_argument('--data_train_path', type=str, default='C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\panel_flt_train.csv')
parser.add_argument('--data_test_path', type=str, default='C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\panel_flt_test.csv')
parser.add_argument('--load_model', type=str, default=None, help='path to saved model')

args = vars(parser.parse_args())

# Pretty print the run args
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

batch_size = args['batch_size']
lr = args['lr']
decay = args['decay']
hidden_dimensions = args['hidden_dimensions']
epochs = args['epochs']
log_interval = args['log_interval']
model_path = args['model_path']
data_train_path = args['data_path']
data_test_path = args['data_path']
load_model = args['load_model']

model_path = f'{model_path}\\lr{lr}\\h{hidden_dimensions}'

try:
    os.makedirs(model_path)
except:
    pass


train_dataset = DatasetMGT(data_train_path)
test_dataset = DatasetMGT(data_test_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


X, y = train_dataset.__getitem__(0)


input_dimensions = X.shape[0]
output_dimensions = 1


model = FcModel(input_dimensions, hidden_dimensions, output_dimensions)

if load_model:
    model.load_state_dict(torch.load(load_model))
    model.eval()

# criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.MSELoss(reduction='mean')
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)


# run the main training loop

losses = []
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        y_pred = model(data)
        loss = criterion(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
            losses.append(loss.data.item())
        if epoch % 500 == 0:
            print(f'Saving model epoch {epoch}')
            torch.save(model.state_dict(), f'{model_path}\\epoch-{epoch}.pt')

plot_loss(losses)
# test model

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
predictions = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data[torch.isnan(data)] = 0
        data, target = Variable(data), Variable(target)
        y_pred = model(data)
        predictions.append(y_pred.item())
        loss = criterion(y_pred[0], target)
        print(loss)

plot_pred(predictions, test_dataset.target.numpy())
