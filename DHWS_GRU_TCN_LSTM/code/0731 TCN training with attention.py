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
from Dilated_TCN_Model import TemporalBlock

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


#准备数据
df = pd.read_csv('data/0728 200user.csv')
train = df[['ls_move', 'p_norm', 'tariff', 'p_flex']][:96*243-96*2]
time_stamp = 96
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data=scaler.fit_transform(train)
x_train, y_train=[], []

for i in range(241-1):
    x_train.append(scaled_data[i*time_stamp:(i+1)*time_stamp,:3])
    y_train.append(scaled_data[(i+1)*time_stamp:(i+2)*time_stamp, 3])
x_train, y_train = np.array(x_train), np.array(y_train)
y_train = y_train.reshape(-1, time_stamp, 1)
sp = int(0.9*len(x_train))
x_train, x_valid, y_train, y_valid = x_train[0:sp], x_train[sp:], y_train[0:sp], y_train[sp:]
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

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
# class TCN(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_channels, device, output_dim, kernel_sz, dropout):
#         super(TCN, self).__init__()
#         self.embed = nn.Linear(input_dim, embed_dim)
#         self.TCNBlock = TemporalConvNet(embed_dim, num_channels, device, kernel_sz, dropout)
#         self.decoder = nn.Linear(num_channels[-1], output_dim)
#
#     def forward(self, x):
#         embeded = self.embed(x)
#         out = self.TCNBlock(embeded.permute(0, 2, 1)).permute(0, 2, 1)
#         out = self.decoder(out)
#
#         return out

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_channels, device, output_dim, kernel_sz, dropout):
        super().__init__()

        assert kernel_sz % 2 == 1, "Kernel size must be odd!"

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.embed = nn.Linear(input_dim, emb_dim)
        self.TCNBlock = TemporalConvNet(emb_dim, num_channels, device, kernel_sz, dropout)
        self.conv2emb = nn.Linear(num_channels[-1], emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):# x=[batch, 96 , inputdim]

        embed = self.embed(x)    # embed = [batch size, 96, emd dim]
        conv_input = embed.permute(0, 2, 1)      # conv_input = [batch size, emb dim, 96]
        conv_out = self.TCNBlock(conv_input ).permute(0,2,1)  # conv_out = [batch size, 96 , num_channels(-1)]
        conved = self.conv2emb(conv_out)     # conved = [batch size, 96, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embed) * self.scale # combined = [batch size, src len, emb dim]

        return conved, combined

# decoder 若也使用dilated_tcn 会造成第i输出考虑过多target,造成cheating
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_layers, kernel_sz, dropout, trg_pad_idx, device):
        super().__init__()

        self.kernel_size = kernel_sz
        self.device = device
        self.trg_pad_idx = trg_pad_idx

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.Linear1 = nn.Linear(output_dim, emb_dim)

        self.attn_emb2out = nn.Linear(emb_dim, output_dim)
        self.attn_out2emb = nn.Linear(output_dim, emb_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=emb_dim, out_channels=2 * emb_dim, kernel_size=kernel_sz) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, output , conved, encoder_conved, encoder_combined):
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, (hid dim)]
        conved_emb = conved.permute(0, 2, 1)

        conved_emb = self.attn_emb2out(conved_emb)  # conved_emb = [batch size, trg len, output dim]

        combined = (conved_emb + output) * self.scale  # combined = [batch size, trg len, output dim]

        combined = self.attn_out2emb(combined)# combined = [batch size, trg len, hid dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, y, encoder_conved, encoder_combined):
        # y = [batch size, 96, output_dim]
        conv_input = self.Linear1(y) # conv_input = [ batch size, 96, emb_dim]
        conv_input = conv_input.permute(0, 2, 1)
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input) # conved = [batch size, hid dim *2, 96]
            conved = F.glu(conved, dim = 1) # conved = [batch size, hid dim, 96]

            # calculate attention
            attention, conved = self.calculate_attention(y, conved, encoder_conved, encoder_combined)
            # conved [batch size, hid dim, trg len]
            # attention = [batch size, trg len, src len]
            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, trg len]
            # set conv_input to conved for next loop iteration
            conv_input = conved

        # conved = [batch size, trg len, hid_dim]

        output = self.fc_out(self.dropout(conved.permute(0,2,1)))

        # output = [batch size, trg len, output dim]

        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encoder_conved, encoder_combined = self.encoder(x)

        output, attention = self.decoder(y, encoder_conved, encoder_combined)

        return output, attention



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
DEC_LAYERS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, EMBED_DIM, channels, device, OUTPUTDIM, KERNEL_SIZE, 0.)
dec = Decoder(OUTPUTDIM, EMBED_DIM, DEC_LAYERS, KERNEL_SIZE, 0., TRG_PAD_IDX, device)

model = Seq2Seq(enc, dec).to(device)



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

        output, _ = model(input, target)

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

            output, _ = model(input, target)
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
MAX_EPOCH = 5

for i in range(MAX_EPOCH):

    train_loss = train(train_loader, model, optimizer, criterion)
    val_loss = eval(val_loader, model, criterion)

    train_loss_plot.append(train_loss)
    val_loss_plot.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '0731 TCN attention model 0.pt')
    if i % 1 == 0:
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        start_time = time.time()
        print('use {} mins {} seconds, epoch: {}, train_loss: {}, val_loss: {}'.format(mins, secs,  i, \
                                                                                       round(train_loss,9), round(val_loss,9)))


np.save('train_loss_plot 0731.npy', train_loss_plot)
np.save('val_loss_plot 0731.npy', val_loss_plot)
total_end_time = time.time()
# total_mins, total_secs = epoch_time(total_start_time, total_end_time)
# print('use {} mins {} secs to train this model in total'.format(total_mins, total_secs))

