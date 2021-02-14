#什么东西阻止了它的梯度下降？

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

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

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

power_agg = torch.Tensor(np.load('power_agg.npy'))
power_agg_LS = torch.Tensor(np.load('power_agg_LS.npy'))
minp = power_agg_LS.min()
maxp = power_agg_LS.max()

power_agg = (power_agg - minp)/(maxp-minp)
power_agg_LS[:,0] = (power_agg_LS[:,0] - minp)/(maxp-minp)

power_agg = power_agg.view(-1, 96, 1)
power_agg_LS = power_agg_LS.view(-1, 96, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 199 * 2

def create_dataset(power_agg, power_agg_LS, inputlen = 96, input_feature = 3,  outlen = 96, outfeature = 1):
    output = power_agg_LS[7:, :, 0:1]
    days = output.shape[0]

    input = torch.zeros(days, inputlen, input_feature)
    input[:,:,0] = power_agg[6:-1, :, 0]
    input[:,:,1] = power_agg[0:-7, :, 0]
    input[:,:,2] = power_agg_LS[7:, :, 1]

    return input, output


input, output = create_dataset(power_agg, power_agg_LS)
print(input.shape)
print(output.shape)

train_ind = int(0.8 * input.shape[0])
val_ind = int(0.9 * input.shape[0])

train_input, train_output = input[:train_ind], output[:train_ind]
train_input, train_output = train_input.repeat(96, 1, 1), train_output.repeat(96, 1, 1)

val_input, val_output = input[train_ind:val_ind], output[train_ind:val_ind]
test_input, test_output = input[val_ind:], output[val_ind:]

print('train_input_sz:{}, train_output_sz:{}\nval_input_sz:{}, val_output_sz:{}\ntest_input_sz:{}, test_output_sz:{}'.\
      format(train_input.shape, train_output.shape, val_input.shape, val_output.shape, test_input.shape, test_output.shape))

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

class TCN(nn.Module):
    def __init__(self,input_dim,output_dim,emb_dim,hid_dim,n_layers,kernel_size,dropout,trg_pad_idx,device,max_length=100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Linear(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.Qlayer = nn.Linear(hid_dim, hid_dim)
        self.Klayer = nn.Linear(hid_dim, hid_dim)
        self.Vlayer = nn.Linear(hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)


    def calculate_attention(self, embedded, conved):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        hid_emb = self.attn_emb2hid(embedded)
        conved = conved.permute(0, 2, 1)
        # conved = [batch size, trg len, hid dim]
        combined = (conved + hid_emb) * self.scale
        Q = self.Qlayer(combined)
        # Q = [batch size, trg len, hid dim]
        K = self.Klayer(combined)
        # K = [batch size, trg len, hid dim]
        V = self.Vlayer(combined)
        # V = [batch size, trg len, hid dim]
        energy = torch.matmul(Q, K.permute(0, 2, 1))
        # energy = [batch size, trg len, trg len]
        attention = F.softmax(energy, dim=2)
        # attention = [batch size, trg len, trg len]
        attended_encoding = torch.matmul(attention, V)
        # attended_encoding = [batch size, trg len, hid dim]
        # apply residual connection
        attended_combined = (conved + attended_encoding) * self.scale
        attended_combined = attended_combined.permute(0, 2, 1)

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, src):
        # src = [batch size, src len, input_dim]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        #embedded = self.dropout(tok_embedded)

        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)

            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, trg len]

            # calculate attention
            # attention, conved = self.calculate_attention(embedded, conved)
            # attention = [batch size, trg len, trg len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        # output = [batch size, trg len, output dim]

        return output

INPUT_DIM = 3
OUTPUTDIM = 1
EMB_DIM = 64
HID_DIM = 128  # each conv. layer has 2 * hid_dim filters
NUM_LAYERS = 10  # number of conv. blocks in encoder
KERNEL_SIZE = 3  # must be odd!
DROPOUT = 0.
TRG_PAD_IDX = 0.

model = TCN(INPUT_DIM, OUTPUTDIM, EMB_DIM, HID_DIM, NUM_LAYERS, KERNEL_SIZE, DROPOUT, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train(train_iterator, model, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for _, (input, target) in enumerate(train_iterator):
        input, target = input.to(device), target.to(device)

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
MAX_EPOCH = 100

for i in range(MAX_EPOCH):

    train_loss = train(train_loader, model, optimizer, criterion)
    val_loss = eval(val_loader, model, criterion)

    train_loss_plot.append(train_loss)
    val_loss_plot.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'flex_pred_origin_model.pt')
    if i%1 == 0:
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        start_time = time.time()
        print('use {} mins {} seconds, epoch: {}, train_loss: {}, val_loss: {}'.format(mins, secs,  i, \
                                                                                       round(train_loss,9), round(val_loss,9)))

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


day = 22 - 1
test_input = test_input.to(device)
model.load_state_dict(torch.load('flex_pred_origin_model.pt'))
model.eval()
predict_power = model(test_input[day:day+1, :, :])
predict_power = predict_power.cpu().detach()

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
