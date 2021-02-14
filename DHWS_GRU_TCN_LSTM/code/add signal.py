import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io



#读入数据
data1 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user1-100.mat')
data2 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user101-150.mat')
data3 = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\user151-200.mat')
tariff = scipy.io.loadmat(r'G:\科研\代码\July_coding\data\Tariff2(1).mat')["Tariff"][0].tolist()
ZP = data1["ZP"].sum(axis=1) + data2["ZP"].sum(axis=1) + data3["ZP"].sum(axis=1)
ZP1 = np.concatenate((data1["ZP"], data2["ZP"][:, 100:], data3["ZP"][:, 150:]), axis=1).T
ZT = np.concatenate((data1["ZTtank"], data2["ZTtank"][:, 100:], data3["ZTtank"][:, 150:]), axis=1).T
pattern = pd.read_csv(r'G:\科研\代码\热水器\data\0519 2000userpattern.csv')
pattern200 = np.array(pattern)[:200, 1:]
tariff1 = []
for i in range(243):
    tariff1 += tariff

print(ZP1.shape)
print(ZT.shape)
print(pattern200.shape)



# 生成ls信号，up down全部覆盖
down = []
for i in range(96):
    row = []
    for j in range(96):
        if j < i:
            row.append(0)
        else:
            row.append(-1)
    down.append(row)
ls_down = []
for i in range(96):
    for j in range(96):
        ls_down.append(down[i][j])

up = []
for i in range(96):
    row = []
    for j in range(96):
        if j < i:
            row.append(0)
        else:
            row.append(1)
    up.append(row)
ls_up = []
for i in range(96):
    for j in range(96):
        ls_up.append(up[i][j])

ls_norm = np.zeros(96, dtype=int).tolist()

ls_all = ls_up + ls_norm + ls_down + ls_norm + ls_up[:49*96]
print(len(ls_all)/96)


# 生成用户行为 1：away;2:sleep;3:bath;4:showering;5:others
# user1=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,4,4,4,4,1,1,1,1,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,3,3,3,3,5,5,5,5,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
# user2=[2,2,2,2,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
# user3=[5,5,5,5,5,5,5,5,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
# user4=[2,2,2,2,2,2,2,2,2,2,2,2,5,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,5,5,5,5,5,5,5,5,4,4,4,4,5,5,5,5,2,2,2,2,2,2,2,2]
# user5=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
#
# pattern_122 = []
# for j in range(2000):
#     pattern = []
#     for i in range(122):
#         x = random.randint(1, 5)
#         if x == 1:
#             pattern.append(user1)
#         if x == 2:
#             pattern.append(user2)
#         if x == 3:
#             pattern.append(user3)
#         if x == 4:
#             pattern.append(user4)
#         if x == 5:
#             pattern.append(user5)
#     pattern = np.array(pattern).reshape(1, 122*96)
#     pattern_122.append(pattern)
# pattern_122 = pd.DataFrame( np.array(pattern_122).reshape(2000,-1))
# pattern_122.to_csv(r'G:\科研\代码\July_coding\data\pattern122days.csv')


# 生成灵活度公式
def flexgenerate(p_base, t_base, dr_sig, userpattern):
    p_flex = np.zeros(96*243, dtype=int)
    for i in range(200):
        flex = np.zeros(96*243, dtype=int).tolist()
        temp = t_base[i]
        p = p_base[i]
        pattern = userpattern[i]
        for j in range(243*96-1):
            if dr_sig[j] == 0:
                flex[j] = 0
            elif dr_sig[j] == -1:
                flex[j] = p[j]
                p[j] = 0
                if pattern[j] == 1 or pattern[j] == 2 or pattern[j] == 5:
                    temp[j+1] = temp[j] - 1.142
                elif pattern[j] == 3:
                    temp[j+1] = temp[j] - 8.14
                else:
                    temp[j+1] = temp[j] - 15.143
                if temp[j] <= 60:
                    flex[j] = 0 # 温控信号与dr信号对冲，此后灵活度均降为零
            else: # 正1信号代表此刻功率变为最大功率
                flex[j] = 3 - p[j]
                p[j] = 3
                if pattern[j] == 1 or pattern[j] == 2 or pattern[j] == 5:
                    temp[j+1] = temp[j] + 6.4
                elif pattern[j] == 3:
                    temp[j+1] = temp[j] - 4.6
                else:
                    temp[j+1] = temp[j] - 8.6
                if temp[j] >= 100:
                    flex[j] = 0 # 温控信号与dr信号对冲，灵活度降为零
        p_flex = p_flex + np.array(flex)
    return p_flex

p_flex = flexgenerate(ZP1, ZT, ls_all, pattern200)

print(len(p_flex))

#数据储存
# data={'p_norm':ZP,'p_flex':p_flex,'ls':ls_all,'tariff':tariff1}
# df=pd.DataFrame(data)
# df.to_csv(r'G:\科研\代码\July_coding\data\0728 200user.csv')



