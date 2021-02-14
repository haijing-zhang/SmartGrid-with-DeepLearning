import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import random
import matplotlib.pyplot as plt2


# data = np.array(pd.read_csv('data/0803 extended data 2.csv')[["ls", "tariff", "p_flex"]])[206*96*2:,]
# power = np.array(pd.read_csv('data/0728 200user.csv')[["p_norm", "tariff"]])[96:, ] # 现在p_norm、ls、p_flex一一对应
# data = np.reshape(data, [-1, 96, 3]) # data = [206*240,96,3]
# data = torch.Tensor(data)
# output = data[:, :, 2]
# LS = data[:, :, : 2]
# print(output.shape, LS.shape)


# data = [206*96*240,3] 206 = 96*2+14
# power = [96*242,2]
# power = torch.Tensor(power)
#     # power = [242*96, 2]
#
#
# input = torch.Tensor([])  # p_norm要比p_flex前移两天
# for i in range(242 - 1):
#     temp = power[i * 96: (i + 2) * 96, :]
#     temp = temp.unsqueeze(0).repeat(206, 1, 1)
#     input = torch.cat((input, temp), dim=0) # input = [240*206, 192 ,2]
# input = input[206:,:,:]
# print(input.shape)

def get_DataLoader(data, power, BATCH_SIZE):

    data = np.reshape(data, [-1, 96, 3]) # data = [206*240,96,3]

    data = torch.Tensor(data)

    for i in range(3): ###归一化
        # data[:, :, i] = (data[:, :, i] - data[:, :, i].mean()) / data[:, :, i].std()
        data[:, :, i] = (data[:, :, i] - data[:, :, i].min()) / (data[:, :, i].max() - data[:, :, i].min())
        print('the max value of dim{} is:{}, the min value is:{}'.format(i, data[:, :, i].max(), data[:, :, i].min()))

    power = (power - power.min()) / (power.max() - power.min())
    power = torch.Tensor(power)
    # power = [242*96, 2]

    input = torch.Tensor([])  # p_norm要比p_flex前移两天
    for i in range(242 - 1):
        temp = power[i * 96: (i + 2) * 96, :]
        temp = temp.unsqueeze(0).repeat(206, 1, 1)
        input = torch.cat((input, temp), dim=0)  # input = [240*206, 192 ,2]
    input = input[206:, :, :]

    output = data[:, :, 2].reshape(-1,96,1)
    LS = data[:, :, : 2]

    train_ind = int(0.8 * input.shape[0])
    val_ind = int(0.9 * input.shape[0])

    train_input, train_output, train_LS = input[:train_ind], output[:train_ind], LS[:train_ind]
    val_input, val_output, val_LS = input[train_ind:val_ind], output[train_ind:val_ind], LS[train_ind:val_ind]
    test_input, test_output, test_LS = input[val_ind:], output[val_ind:], LS[val_ind:]
    print('train_input.shape: {}, train_output.shape: {}, \nval_input.shape: {}, val_output.shape: {}'. \
          format(train_input.shape, train_output.shape, val_input.shape, val_output.shape))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, input, output, LS):
            self.input = input
            self.output = output
            self.LS = LS

        def __getitem__(self, index):
            return self.input[index], self.output[index], self.LS[index]

        def __len__(self):
            return self.input.shape[0]

    train_dataset = Dataset(input=train_input, output=train_output, LS=train_LS)
    val_dataset = Dataset(val_input, val_output, val_LS)

    train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE)

    return train_loader, val_loader, test_input, test_output, test_LS

# batch_sz = 192 * 1
# # load data
# train_loader, val_loader, test_input, test_output, test_LS = get_DataLoader(data, power, batch_sz)
# print(train_loader)
