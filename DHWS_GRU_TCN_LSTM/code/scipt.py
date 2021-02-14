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

real_flex_up = []
for i in range(29):
    flex1 = []
    for j in range(96):
        p = ZP1[:, 243*96-96*(29-i) : 243*96-96*(28-i)].copy()# 这一步很关键，否则函数会直接对ZP1进行修改，无法循环
        t = ZT[:, 243*96-96*(29-i) : 243*96-96*(28-i)].copy()# 同上
        a = up[j]
        flex = flexoneday(p, t , a, pattern200[:,243*96-96*(29-i) : 243*96-96*(28-i)]).tolist()
        flex1.append(flex)
    real_flex_up.append(flex1)

test_up =[]
for i in range(29):
    test = []
    for j in range(96):
        a=np.hstack((np.array(scaled_up[j]).reshape(96,1),x_test[i,:,1:]))
        test.append(a)
test_up = np.array(test)


test_down =[]
for i in range(96):
    a=np.hstack((np.array(scaled_down[i]).reshape(96,1),x_test))
    test_down.append(a)
test_down = np.array(test_down)

pre_flex_up=[]
for i in range(29):
    pre = []
    for i in range(96):
        a = torch.Tensor(test_up[i,j,:,:].reshape(1,96,3))
        pre_flex = model(a)
        scaler.fit_transform(df['p_flex'].values.reshape(-1,1))
        pre_flex = scaler.inverse_transform(pre_flex.view(1,96).detach().numpy())
        pre.append(pre_flex)
    pre_flex_up.append(pre)
pre_flex_up = np.array(pre_flex_up)

ds_0 = np.zeros(96, dtype=int).tolist()
ext_pbase = []
ext_ds = []
ext_pflex = []
ext_tariff = []
for i in range(243):
    for j in range(96):
        p = ZP1[:, i * 96:(i + 1) * 96].copy()
        t = ZP1[:, i * 96:(i + 1) * 96].copy()
        a = up[j]
        pattern = pattern200[:, i * 96:(i + 1) * 96]
        flex = flexoneday(p, t, a, pattern).tolist
        ext_pflex += flex
        ext_pbase += p_base[i]
        ext_tariff += tariff
        ext_ds += a
    for j in range(96):
        p = ZP1[:, i * 96:(i + 1) * 96].copy()
        t = ZP1[:, i * 96:(i + 1) * 96].copy()
        a = down[j]
        pattern = pattern200[:, i * 96:(i + 1) * 96]
        flex = flexoneday(p, t, a, pattern).tolist
        ext_pflex += flex
        ext_pbase += p_base[i]
        ext_tariff += tariff
        ext_ds += a
    for j in range(14):
        p = ZP1[:, i * 96:(i + 1) * 96].copy()
        t = ZP1[:, i * 96:(i + 1) * 96].copy()
        pattern = pattern200[:, i * 96:(i + 1) * 96]
        flex = flexoneday(p, t, ds_0, pattern).tolist
        ext_pflex += flex
        ext_pbase += p_base[i]
        ext_tariff += tariff
        ext_ds += ds_0


