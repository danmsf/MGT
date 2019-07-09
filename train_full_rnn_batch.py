from mgtModels import FullLSTM
import torch
from torch.utils.data import DataLoader
from dataset import DatasetLstmFullAll
from torch.autograd import Variable
from mgtUtils import plot_loss
import os

lr = 1e-06
decay = 0.9
batch_size = 10
num_epochs = 1
load_model = False
batchpath = "C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\batchlist.txt"
jsonpath = 'C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\labels.json'
saveloss = 'C:\\Users\\Dan\\PycharmProjects\\MGT\\data\\saveloss.txt'
model_path = 'C:\\Users\\Dan\\PycharmProjects\\MGT\\saved_models\\fullrnn\\'

train_dataset = DatasetLstmFullAll(jsonpath, batchpath)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

x, y, x_length = train_dataset.__getitem__(10)
input_dimension = x.transpose(0, 1).shape[1]

model = FullLSTM(input_dimensions=input_dimension, batch_size=batch_size, nb_layers=1)

load_batch = 0
if load_model:
    load_batch = 1303
    model.load_state_dict(torch.load(os.path.join(model_path, f'batch-{load_batch}.pt')))
    model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
loss = model.criterion
losses = []
for e in range(num_epochs):
    for batch_idx, (data, target, x_length) in enumerate(train_loader):
        print(batch_idx)
        data = data.transpose(1, 2)
        data, target = Variable(data), Variable(target)
        yhat = model(data, x_length)
        # loss, yhat = model.loss(yhat, torch.Tensor(target), seqs_len)
        loss = model.criterion(yhat.squeeze(1), target.to(torch.float32))
        losses.append(loss.item())
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 :
            print('loss {}'.format(loss))
        if batch_idx % 100 == 0 :
            with open(saveloss, "w") as f:
                for item in losses:
                    f.write("%s," % item)
                    torch.save(model.state_dict(), os.path.join(model_path, f'batch-{batch_idx + 1 + load_batch}.pt'))
plot_loss(losses)
