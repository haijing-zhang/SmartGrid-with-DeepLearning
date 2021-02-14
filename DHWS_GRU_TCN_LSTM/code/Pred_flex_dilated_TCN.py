
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import pandas as pd


# import Dilated_TCN_Model
from Dilated_TCN_Model import TemporalConvNet

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

#创建数据集
# power_agg = torch.Tensor(np.load('power_agg.npy'))
# power_agg_LS = torch.Tensor(np.load('power_agg_LS.npy'))
# 做归一化
# minp = power_agg_LS.min()
# maxp = power_agg_LS.max()
#
# power_agg = (power_agg - minp)/(maxp-minp)
# power_agg_LS[:,0] = (power_agg_LS[:,0] - minp)/(maxp-minp)
#
# power_agg = power_agg.view(-1, 96, 1)
# power_agg_LS = power_agg_LS.view(-1, 96, 2)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

BATCH_SIZE = 199 * 2

# def create_dataset(power_agg, power_agg_LS, inputlen = 96, input_feature = 3,  outlen = 96, outfeature = 1):
#     output = power_agg_LS[7:, :, 0:1]
#     days = output.shape[0]
#
#     input = torch.zeros(days, inputlen, input_feature)
#     input[:,:,0] = power_agg[6:-1, :, 0]
#     input[:,:,1] = power_agg[0:-7, :, 0]
#     input[:,:,2] = power_agg_LS[7:, :, 1]
#
#     return input, output
#
#
# input, output = create_dataset(power_agg, power_agg_LS)
# print(input.shape)
# print(output.shape)
#
# train_ind = int(0.8 * input.shape[0])
# val_ind = int(0.9 * input.shape[0])
#
# train_input, train_output = input[:train_ind], output[:train_ind]
# train_input, train_output = train_input.repeat(96, 1, 1), train_output.repeat(96, 1, 1)
#
# val_input, val_output = input[train_ind:val_ind], output[train_ind:val_ind]
# test_input, test_output = input[val_ind:], output[val_ind:]
#
# print('train_input_sz:{}, train_output_sz:{}\nval_input_sz:{}, val_output_sz:{}\ntest_input_sz:{}, test_output_sz:{}'.\
#       format(train_input.shape, train_output.shape, val_input.shape, val_output.shape, test_input.shape, test_output.shape))



class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return self.input.shape[0]

train_dataset = Dataset(input=train_input, output=train_output)
val_dataset = Dataset(val_input, val_output)

train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE)


#定义模型
class TCN(nn.Module):
    def __init__(self, input_dim, embed_dim, num_channels, device, output_dim, kernel_sz, dropout):
        super(TCN, self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.TCNBlock = TemporalConvNet(embed_dim, num_channels, device, kernel_sz, dropout)
        self.decoder = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        embeded = self.embed(x)
        out = self.TCNBlock(embeded.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.decoder(out)

        return out

# 计算最远可以看到多远
channels = [128, 128, 128, 128, 128, 128]
KERNEL_SIZE = 3
dependency = KERNEL_SIZE * len(channels) * (len(channels) + 1) - len(channels)
print('Now your model can capture {} step dependency'.format(dependency))

# 模型实例化
model = TCN(INPUT_DIM, EMBED_DIM, channels, device, OUTPUTDIM, KERNEL_SIZE, 0.).to(device)

#模型需计算的参数数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# 参数预设
INPUT_DIM = 3
EMBED_DIM = 64
OUTPUTDIM = 1
DROPOUT = 0.
TRG_PAD_IDX = 0.
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# 训练模型
def train(train_iterator, model, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for _, (input, target) in enumerate(train_iterator):
        input, target = input.to(device), target.to(device)
        input = input.contiguous()

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        epoch_loss += loss.cpu().item()

    return epoch_loss/(len(train_iterator))

def eval(eval_iterator, model, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (input, target) in enumerate(eval_iterator):
            input, target = input.to(device), target.to(device)
            input = input.contiguous()

            output = model(input)
            epoch_loss += criterion(output, target).cpu().item()

    return epoch_loss/(len(eval_iterator))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

train_loss_plot = []
val_loss_plot = []
best_val_loss = float('inf')

start_time = time.time()
total_start_time = start_time
MAX_EPOCH = 40

# for i in range(MAX_EPOCH):
#
#     train_loss = train(train_loader, model, optimizer, criterion)
#     val_loss = eval(val_loader, model, criterion)
#
#     train_loss_plot.append(train_loss)
#     val_loss_plot.append(val_loss)
#
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'flex_pred_origin_model.pt')
#     if i % 1 == 0:
#         end_time = time.time()
#         mins, secs = epoch_time(start_time, end_time)
#         start_time = time.time()
#         print('use {} mins {} seconds, epoch: {}, train_loss: {}, val_loss: {}'.format(mins, secs,  i, \
#                                                                                        round(train_loss,9), round(val_loss,9)))

np.save('train_loss_plot.npy', train_loss_plot)
np.save('val_loss_plot.npy', val_loss_plot)
total_end_time = time.time()
total_mins, total_secs = epoch_time(total_start_time, total_end_time)
print('use {} mins {} secs to train this model in total'.format(total_mins, total_secs))

##########end training

train_loss_plot = np.load('train_loss_plot.npy')
val_loss_plot = np.load('val_loss_plot.npy')
train_loss_plot = np.reshape(train_loss_plot, [MAX_EPOCH, -1])
val_loss_plot = np.reshape(val_loss_plot, [MAX_EPOCH, -1])

# loss_plot = np.zeros([MAX_EPOCH, 2])
# loss_plot[:, 0] = train_loss_plot[:, 0]
# loss_plot[:, 1] = val_loss_plot[:, 0]
#
# exceldata = pd.DataFrame(loss_plot)
# exceldata.to_excel('loss curve.xlsx')


# day = 22 - 1
# test_input = test_input.to(device)
# model.load_state_dict(torch.load('flex_pred_origin_model.pt'))
# model.eval()
# predict_power = model(test_input[day:day+1, :, :])
# predict_power = predict_power.cpu().detach()

plt.figure()
plt.plot(test_output[day, :, 0])
plt.hold
plt.plot(predict_power[0, :, 0])

plt.figure()
plt.plot(train_loss_plot)
plt.hold
plt.plot(val_loss_plot)
plt.legend(['train_loss_curve', 'val_loss_curve'])

train_loss_plot = np.array(train_loss_plot)
val_loss_plot = np.array(val_loss_plot)