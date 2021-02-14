import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


#准备数据
# data_input = np.load('data/input_features.npy')
# data_output = np.load('data/p_flex.npy')
df = pd.read_csv('data/0802 extended data.csv')
data_input = np.array(df[["ls","p_norm","tariff"]])
data_output = np.array(df['p_flex']).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_input = scaler.fit_transform(data_input)
scaled_output = scaler.fit_transform(data_output)

time_stamp = 96
scaled_train_in = scaled_input[:-96*30]
scaled_train_out = scaled_output[:-96*30]
scaled_test_in = scaled_input[-96*30:]
scaled_test_out = scaled_output[-96*30:]

x_train, y_train=[], []
for i in range(int(len(scaled_train_in)/time_stamp)):
    x_train.append(scaled_train_in[i*time_stamp:(i+1)*time_stamp,:3])
    y_train.append(scaled_train_out[i*time_stamp:(i+1)*time_stamp, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
y_train = y_train.reshape(-1, time_stamp, 1)
sp = int(0.8*len(x_train))
x_train, x_valid, y_train, y_valid = x_train[0:sp], x_train[sp:], y_train[0:sp], y_train[sp:]
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

x_test, y_test=[], []
for i in range(int(len(scaled_test_in)/time_stamp)):
    x_test.append(scaled_test_in[i*time_stamp:(i+1)*time_stamp,:3])
    y_test.append(scaled_test_out[i*time_stamp:(i+1)*time_stamp, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_test = y_test.reshape(-1,time_stamp,1)
print(x_test.shape, y_test.shape)
np.save('0803 extened_x_test.npy',x_test)
np.save('0803 extened_y_test.npy',y_test)

# 创建数据集
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_valid = torch.Tensor(x_valid)
y_valid = torch.Tensor(y_valid)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, index):
        return self.input[index], self.output[index]

    def __len__(self):
        return self.input.shape[0]

train_dataset = Dataset(input= x_train, output= y_train)
val_dataset = Dataset(input= x_valid, output= y_valid)
BATCH_SIZE = 199 * 2
train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE)

# 创建模型
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

# 参数预设
INPUT_DIM = 3
EMBED_DIM = 64
OUTPUTDIM = 1
DROPOUT = 0.
TRG_PAD_IDX = 0.


# 模型实例化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(INPUT_DIM, EMBED_DIM, channels, device, OUTPUTDIM, KERNEL_SIZE, 0.).to(device)

# 模型需计算的参数数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


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
MAX_EPOCH = 2

for i in range(MAX_EPOCH):

    train_loss = train(train_loader, model, optimizer, criterion)
    val_loss = eval(val_loader, model, criterion)

    train_loss_plot.append(train_loss)
    val_loss_plot.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '0803 TCN model test.pt')
    if i % 1 == 0:
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        start_time = time.time()
        print('use {} mins {} seconds, epoch: {}, train_loss: {}, val_loss: {}'.format(mins, secs,  i, \
                                                                                       round(train_loss,9), round(val_loss,9)))

np.save('0803 TCN train_loss_plot.npy', train_loss_plot)
np.save('0803 TCN val_loss_plot.npy', val_loss_plot)
total_end_time = time.time()
# total_mins, total_secs = epoch_time(total_start_time, total_end_time)
# print('use {} mins {} secs to train this model in total'.format(total_mins, total_secs))

