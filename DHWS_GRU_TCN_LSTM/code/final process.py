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
import scipy.io



df = pd.read_csv(r'G:\科研\代码\July_coding\data\0728 200user.csv')
data1 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user1-100.mat')
data2 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user101-150.mat')
data3 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user151-200.mat')
tariff = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\Tariff2(1).mat')["Tariff"][0].tolist()
ZP = data1["ZP"].sum(axis=1) + data2["ZP"].sum(axis=1) + data3["ZP"].sum(axis=1)
ZP1 = np.concatenate((data1["ZP"], data2["ZP"][:, 100:], data3["ZP"][:, 150:]), axis=1).T
ZT = np.concatenate((data1["ZTtank"], data2["ZTtank"][:, 100:], data3["ZTtank"][:, 150:]), axis=1).T
# pattern = pd.read_csv(r'G:\科研\代码\热水器\data\0519 2000userpattern.csv')
# pattern200 = np.array(pattern)[:200, 1:]
pattern200 = np.array(pd.read_csv(r'G:\科研\代码\热水器\data\pattern200.csv'))[:, 1:]
print(pattern200.shape)

#生成测试数据
test=df[['p_norm','tariff']]
scaler = MinMaxScaler(feature_range=(0, 1))
x_test=scaler.fit_transform(test[96*243-96*2:96*243-96*1])
real_test=df[['p_norm']][96*243-96*1:]

#生成up和down不同的96组信号

down = []
for i in range(96):
    row = []
    for j in range(96):
        if j < i:
            row.append(0)
        else:
            row.append(-1)
    down.append(row)
down = np.array(down)

up = []
for i in range(96):
    row = []
    for j in range(96):
        if j < i:
            row.append(0)
        else:
            row.append(1)
    up.append(row)
up = np.array(up)


#修改成只生成最后一天的函数
def flexoneday(p_base, t_base, dr_sig, userpattern):
    p_flex = np.zeros(96 , dtype=int)
    for i in range(200):
        flex = np.zeros(96, dtype=int).tolist()
        temp = t_base[i]
        p = p_base[i]
        pattern = userpattern[i]
        for j in range(96 - 1):
            if dr_sig[j] == 0:
                flex[j] = 0
            elif dr_sig[j] == -1:
                flex[j] = p[j]
                p[j] = 0
                if pattern[j] == 1 or pattern[j] == 2 or pattern[j] == 5:
                    temp[j + 1] = temp[j] - 1.142
                elif pattern[j] == 3:
                    temp[j + 1] = temp[j] - 8.14
                else:
                    temp[j + 1] = temp[j] - 15.143
                if temp[j] <= 60:
                    flex[j] = 0  # 温控信号与dr信号对冲，此后灵活度均降为零
            else:  # 正1信号代表此刻功率变为最大功率
                flex[j] = 3 - p[j]
                p[j] = 3
                if pattern[j] == 1 or pattern[j] == 2 or pattern[j] == 5:
                    temp[j + 1] = temp[j] + 6.4
                elif pattern[j] == 3:
                    temp[j + 1] = temp[j] - 4.6
                else:
                    temp[j + 1] = temp[j] - 8.6
                if temp[j] >= 100:
                    flex[j] = 0  # 温控信号与dr信号对冲，灵活度降为零
        p_flex = p_flex + np.array(flex)
    return p_flex

print(ZP1[:,243*96-96:].shape, ZT[:,243*96-96:].shape, pattern200[:,243*96-96:].shape)


real_flex_up = []
for i in range(96):
    p=ZP1[:,243*96-96:].copy()
    t=ZT[:,243*96-96:].copy()
    a=up[i]
    flex = flexoneday(p, t , a, pattern200[:,243*96-96:]).tolist()
    real_flex_up.append(flex)
print(real_flex_up)
