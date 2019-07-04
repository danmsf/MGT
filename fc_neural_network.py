import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DatasetMGT
from torch.autograd import Variable
import os
from mgtModels import FcModel
from mgtUtils import *

batch_size = 271
lr = 1e-6
decay = 0.9
hidden_dimenssions = 800
epochs = 3000
log_interval = 10

model_path = 'C:\\Users\\Dan\\PycharmProjects\\MGT\\saved_models'
model_path = f'{model_path}\\lr{lr}\\h{hidden_dimenssions}'

train_dataset = DatasetMGT('C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\panel_flt_train.csv')
test_dataset = DatasetMGT('C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\panel_flt_test.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


X, y = train_dataset.__getitem__(0)


input_dimensions = X.shape[0]
output_dimenssions = 1


try:
    os.makedirs(model_path)
except:
    pass

# Create random Tensors to hold inputs and outputs.
# x = torch.randn(batch_size, input_dimensions)
# y = torch.randn(batch_size, output_dimensions)


model = FcModel(input_dimensions, hidden_dimenssions, output_dimenssions)

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


# Load model
# model = TheModelClass(*args, **kwargs)
model = Model(input_dimensions, hidden_dimenssions, output_dimenssions)
model.load_state_dict(torch.load(f'{model_path}\\epoch-500.pt'))
model.eval()