
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import pandas as pd

import sys

from OptimalSena7daysInput.Models_for_7days.Model import TemporalConvNet, Encoder, Decoder, Transformer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

from DHW_create_dataset import get_DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
## data.npy是只维持4个步长的，optSOCdata.npy是全发1的
data = np.array(pd.read_csv('data/0803 extended data 2.csv')[["ls", "tariff", "p_flex"]])[206*96*2:,]
power = np.array(pd.read_csv('data/0728 200user.csv')[["p_norm", "tariff"]])[96:, ]
batch_sz = 192 * 1
# load data
train_loader, val_loader, test_input, test_output, test_LS = get_DataLoader(data, power, batch_sz)

INPUT_DIM = 2 # p_norm tariff
CNN_Channels = [64, 64, 64, 64]
Kernel_SZ = 4
OUTPUT_DIM = 1 # p_flex
HID_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 256
DEC_PF_DIM = 256
CNN_DROPOUT = 0.1
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

down_sample_cnn = TemporalConvNet(INPUT_DIM,
                                  CNN_Channels,
                                  device,
                                  Kernel_SZ,
                                  CNN_DROPOUT)

enc = Encoder(CNN_Channels[-1],
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = 10
TRG_PAD_IDX = 10

model = Transformer(down_sample_cnn, enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

def train(train_iterator, model, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for _, (input, target, LS) in enumerate(train_iterator):
        input, target, LS = input.to(device), target.to(device), LS.to(device)
        padding = torch.zeros(target.shape[0], 1, target.shape[2]).to(device)
        target = torch.cat((padding, target), dim=1)

        padding = torch.zeros(target.shape[0], 1, LS.shape[2]).to(device)
        LS = torch.cat((LS, padding), dim=1)

        output, attention = model(input, torch.cat((target, LS), dim=2)[:, :-1, :])
        loss = criterion(output, target[:, 1:, :])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)

        optimizer.step()

        epoch_loss += loss.cpu().item()

    return epoch_loss/(len(train_iterator))

def eval(eval_iterator, model, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (input, target, LS) in enumerate(eval_iterator):
            input, target, LS = input.to(device), target.to(device), LS.to(device)
            padding = torch.zeros(target.shape[0], 1, target.shape[2]).to(device)
            target = torch.cat((padding, target), dim=1)

            padding = torch.zeros(target.shape[0], 1, LS.shape[2]).to(device)
            LS = torch.cat((LS, padding), dim=1)


            output, attention = model(input, torch.cat((target, LS), dim=2)[:, :-1, :])
            epoch_loss += criterion(output, target[:, 1:, :]).cpu().item()

    return epoch_loss/(len(eval_iterator))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

total_start_time = time.time()
MAX_EPOCH = 2
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500, gamma=0.2, last_epoch=-1)


def START_TRAINING():
    train_loss_plot = []
    val_loss_plot = []
    start_time = time.time()
    best_val_loss = float('inf')
    for i in range(MAX_EPOCH):

        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss = eval(val_loader, model, criterion)
        scheduler.step()

        train_loss_plot.append(train_loss)
        val_loss_plot.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_Models/0803 Transformer_model.pt')
        if i % 1 == 0:
            end_time = time.time()
            mins, secs = epoch_time(start_time, end_time)
            start_time = time.time()
            print('use {} mins {} seconds, epoch: {}, lr: {}, train_loss: {}, val_loss: {}'.format(mins, secs, i, \
                                                                                                   scheduler.get_lr()[0],
                                                                                                   round(train_loss, 7),
                                                                                                   round(val_loss, 7)))
        np.save('./loss_curve/loss_curve_for_Transformer/0803 train_loss_plot.npy', train_loss_plot)
        np.save('./loss_curve/loss_curve_for_Transformer/0803 val_loss_plot.npy', val_loss_plot)

START_TRAINING()  #train the network

total_end_time = time.time()
total_mins, total_secs = epoch_time(total_start_time, total_end_time)
print('use {} mins {} secs to train this model in total'.format(total_mins, total_secs))

##########end training

# train_loss_plot = np.load('./loss_curve/loss_curve_for_Transformer/train_loss_plot.npy')
# val_loss_plot = np.load('./loss_curve/loss_curve_for_Transformer/val_loss_plot.npy')
# train_loss_plot = np.reshape(train_loss_plot, [-1, 1])
# val_loss_plot = np.reshape(val_loss_plot, [-1, 1])
#
# # loss_plot = np.zeros([MAX_EPOCH, 2])
# # loss_plot[:, 0] = train_loss_plot[:, 0]
# # loss_plot[:, 1] = val_loss_plot[:, 0]
# #
# # exceldata = pd.DataFrame(loss_plot)
# # exceldata.to_excel('loss curve.xlsx')
#
# def translate_sentence(src, trg, LS, model, device, max_len=96):
#     model.eval()
#
#     src_tensor = src.to(device)
#
#     src_mask = model.make_src_mask(src_tensor)
#
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)
#
#     trg_indexes = [[0, LS[0, 0, 0]]]
#
#     for i in range(max_len):
#
#         trg_tensor = torch.Tensor(trg_indexes).unsqueeze(0).to(device)
#
#         trg_mask = model.make_trg_mask(trg_tensor)
#
#         with torch.no_grad():
#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
#
#         pred_token = output[0, -1, -1].item()
#
#         if i < max_len-1:
#             trg_indexes.append([pred_token, LS[0, i+1, 0]])
#         else:
#             trg_indexes.append([pred_token, LS[0, i, 0]])
#
#     return torch.Tensor(trg_indexes).numpy()[1:, 0], attention
#
# day = 4000 - 1
# test_input = test_input.to(device)
# test_output = test_output.to(device)
# test_LS = test_LS.to(device)
#
# model.load_state_dict(torch.load('saved_Models/Transformer_model_5e-5.pt'))
# model.eval()
#
# src, trg, LS = test_input[day:day+1, :, :], test_output[day:day+1, :, :], test_LS[day:day+1, :, :]
# predict_power, _ = model(test_input[day:day+1, :, :], torch.cat((test_output[day:day+1, :, :], test_LS[day:day+1, :, :]), dim=2)[:, :-1, :])
#
# predict_power, attention = translate_sentence(src, trg, LS, model, device)
#
# test_output = test_output.cpu()
# test_LS = test_LS.cpu()
# plt.figure()
# plt.plot(test_output[day, :, 0])
# plt.hold
# plt.plot(predict_power)
# # plt.plot(test_LS[day, :, 0])
#
# plt.legend(['true power', 'predicted power'])
#
# plt.figure()
# plt.plot(train_loss_plot)
# plt.hold
# plt.plot(val_loss_plot)
# plt.legend(['train_loss_curve', 'val_loss_curve'])
#
# train_loss_plot = np.array(train_loss_plot)
# val_loss_plot = np.array(val_loss_plot)