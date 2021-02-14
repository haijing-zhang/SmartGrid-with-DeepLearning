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

from DHW_create_dataset2 import get_DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
## data.npy是只维持4个步长的，optSOCdata.npy是全发1的
data = np.array(pd.read_csv('data/0803 extended data 2.csv')[["ls", "tariff", "p_flex"]])[206*96*2:,]
power = np.array(pd.read_csv('data/0728 200user.csv')[["p_norm", "tariff"]])[96:, ]
batch_sz = 192 * 1
# load data
train_loader, val_loader, test_input, test_output, test_LS = get_DataLoader(data, power, batch_sz)

test_input = test_input[-30*206:, :, :]
test_output = test_output[-30*206:, :, :]
test_LS = test_LS[-30*206:, :, :]

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

enc = Encoder(INPUT_DIM,
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

model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load ('models/0803 Transformer_model e-03.pt', map_location='cpu'))
def translate_sentence(src, LS, model, max_len=96):
    model.eval()
    LS = LS.to(device)
    src_tensor = src.to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = torch.zeros(src.shape[0], 1, 1).to(device)

    for i in range(max_len):

        trg_tensor = torch.cat((trg_indexes, LS[:, :i+1, :]), dim=2)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        trg_indexes = torch.cat((trg_indexes, output[:, -1:, :]), dim=1)

    return trg_indexes[:, 1:, :], attention

cur_times = 10  #拆开数据后的计算次数，直接计算会显存不足
batch = test_input.shape[0]/cur_times #注意一定要能被整除
batch = int(batch)

predicted_output = torch.Tensor([])
for i in range(cur_times):

    temp, _ = translate_sentence(torch.Tensor(test_input[batch * i:batch * (i+1)]),\
                              torch.Tensor(test_LS[batch * i:batch * (i+1)]),\
                              model)
    predicted_output = torch.cat((predicted_output, temp.cpu().detach()), dim=0)

predicted_output = predicted_output.numpy()
