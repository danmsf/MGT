import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetMGT(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.len = len(self.data)
        self.inputs = torch.from_numpy(self.data.drop(columns='target').values).to(torch.float32)
        self.target = torch.from_numpy(self.data.target.values).to(torch.float32)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        return self.inputs[index], self.target[index]



class DatasetLstmMGT(Dataset):

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)
        self.inputs, self.target = self.prep_data()
        self.len = self.inputs.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.inputs[:, index , :], self.target[index]

    def prep_data(self):
        panel = self.data
        panel.rename(columns={"MEAN":"target", "Unnamed: 0":"BATCH_VAL0"}, inplace=True)
        panel['Stage'] = panel['STATESTR_VAL0'] \
                            .map({"Pre-Mixing": 0, "Heating1": 1, "Hot1": 2})
        panel.drop(columns = 'STATESTR_VAL0', inplace = True)

        n_dim = panel.shape[1] - 3
        panel_tensor = torch.Tensor(n_dim, 1, 3)
        target_tensor = torch.FloatTensor(1)
        for key, data in panel.groupby('BATCH_VAL0'):
            if data.shape[0] == 3:
                target_tensor = torch.cat((target_tensor, torch.FloatTensor([data['target'].values[0]])), 0)
                panel_tensor = torch.cat((panel_tensor, torch.from_numpy(
                                                            data.sort_values(['Stage']) \
                                                            .drop(columns= ['Stage', 'BATCH_VAL0', 'target']).values.T
                                                            ).to(torch.float32) \
                                                            .view(-1, 1, 3)), 1)
        panel_tensor = panel_tensor[:, 1:, :]
        target_tensor = target_tensor[1:]
        return panel_tensor, target_tensor

