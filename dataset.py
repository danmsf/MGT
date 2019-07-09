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


import json
import pandas as pd
import pyodbc

class DatasetLstmFull(Dataset):

    def __init__(self, labelspath, batchlist, transform=None):
        self.db = pyodbc.connect('Driver={SQL Server};'
                        'UID=Dan;'
                        'Server=localhost;'
                        'Database=Cimplicity;'
                        'PWD=D@2019;'
                        'Trusted_Connection=no;')
        self.listIDs, self.target = self.__getindexlabels__(labelspath, batchlist)
        self.len = self.__len__()

    def __len__(self):
        return len(self.listIDs)

    def __getindexlabels__(self, jsonpath, batchlist):
        with open(jsonpath, "r") as file:
            labels = json.load(file)

        with open(batchlist, "r") as file:
            index_list = file.read().split('\n')
        return index_list[:-2], labels

    def __getitem__(self, index):
        ID = self.listIDs[index]
        query = "SELECT BATCH_VAL0, STATESTR_VAL0, timeFromStartBatch, Temp1, Temp11, Temp2, Temp6, Temp7, Temp8, RPM1, RPM2, RPM3, Curr3, Prss1, WT1, WT2" + \
                " FROM [dbo].[GSTAT_CYAN_CTR_Final] WHERE STATESTR_VAL0 IN ('Pre-Mixing','Heating1','Hot1') AND BATCH_VAL0 IN {} ".format(tuple([ID]+['0']))
        inputs = pd.read_sql(query, self.db)
        inputs = torch.from_numpy(inputs \
                                .sort_values(['timeFromStartBatch']) \
                                .drop(columns=['STATESTR_VAL0', 'BATCH_VAL0','timeFromStartBatch']) \
                                .values.T).to(torch.float32).view(-1, 1, inputs.shape[0])
        x_length = inputs.shape[2]
        inputs_pad = torch.zeros(size=(inputs.shape[0], inputs.shape[1], 3000))
        inputs_pad[:,:,:x_length] = inputs
        target = torch.Tensor([self.target[ID]])
        return inputs_pad, target, x_length


class DatasetLstmFullAll(Dataset):

    def __init__(self, labelspath, batchlist, transform=None):
        self.db = pyodbc.connect('Driver={SQL Server};'
                        'UID=Dan;'
                        'Server=localhost;'
                        'Database=Cimplicity;'
                        'PWD=D@2019;'
                        'Trusted_Connection=no;')
        self.listIDs, self.targetdict = self.__getindexlabels__(labelspath, batchlist)
        self.len = self.__len__()
        self.input, self.target, self.input_length, self.target_mean, self.target_std = self.__loaddata__()

    def __len__(self):
        return len(self.listIDs)

    def __getindexlabels__(self, jsonpath, batchlist):
        with open(jsonpath, "r") as file:
            labels = json.load(file)

        with open(batchlist, "r") as file:
            index_list = file.read().split('\n')
        return index_list[:-2], labels

    def __loaddata__(self):
        query = "SELECT BATCH_VAL0, STATESTR_VAL0, timeFromStartBatch, Temp1, Temp11, Temp2, Temp6, Temp7, Temp8, RPM1, RPM2, RPM3, Curr3, Prss1, WT1, WT2" + \
                " FROM [dbo].[GSTAT_CYAN_CTR_Final] WHERE STATESTR_VAL0 IN ('Pre-Mixing','Heating1','Hot1') AND BATCH_VAL0 IN {} ".format(
                    tuple(self.listIDs))
        data = pd.read_sql(query, self.db)
        data_list = []
        data_length = []
        target_list = []
        for b, tbl in data.groupby('BATCH_VAL0'):
            tbl = torch.from_numpy(tbl \
                                      .sort_values(['timeFromStartBatch']) \
                                      .drop(columns=['STATESTR_VAL0', 'BATCH_VAL0', 'timeFromStartBatch']) \
                                      .values.T).to(torch.float32).view(-1, tbl.shape[0])
            x_length = tbl.shape[1]
            tbl_pad = torch.zeros(size=(tbl.shape[0], 3000))
            tbl_pad[:, :x_length] = tbl
            data_list.append(tbl_pad)
            target_list.append(self.targetdict[str(b)])
            data_length.append(x_length)
        target_mean = np.mean(target_list)
        target_std = np.std(target_list)
        target_list = [(m-target_mean)/target_std for m in target_list]
        return data_list, target_list, data_length, target_mean, target_std

    def __getitem__(self, index):
        return self.input[index], self.target[index], self.input_length[index]
