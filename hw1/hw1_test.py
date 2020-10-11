import sys
import pandas as pd
import numpy as np
import math
import csv

# 读取文件，文件为4320*24的数据，18个feature，每天24小时的数据
data = pd.read_csv('train.csv', encoding='big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

"""

    提取特征
"""
# 将12个月的数据转化为18(features)*480(hours), 每月前20天，每天24小时
# month_data为长度为12的字典，每个元素为18*480的矩阵
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day*24: (day+1)*24] = raw_data[18 * (20*month + day): 18 * (20*month + day + 1), :]
    month_data[month] = sample

# 每9小时形成一组data，一个月共471个data（即第十小时的数据组为预测值）
# x的行代表12个月*每个月471个数据，列代表18*9个特征（即每小时有18个特征，9个小时的特征单独计算）
# y为行代表471*12个数据总数，列代表一个feature（第十小时的PM2.5值）
x = np.empty([12*471, 18*9], dtype=float)
y = np.empty([12*471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            # 20号及之后的数据没有，hour最大为14时，hour+9=23时
            if day == 19 and hour > 14:
                continue
            # 后面代表每9个小时，reshape(1, -1)代表作为一行数据
            x[month*471 + day*24 + hour, :] = month_data[month][:, day*24 + hour: day*24 + hour + 9].reshape(1, -1)
            # 前一个9是第九行代表特征值PM2.5，后一个代表第十个小时
            y[month*471 + day*24 + hour, 0] = month_data[month][9, day*24 + hour + 9]

"""
    特征归一化
    不同特征值的等比例放缩，避免一个特征值影响过大
"""
# x的列求平均值，即12*471个总数据的平均值，最后留下18*9个特征
mean_x = np.mean(x, axis=0)
std_x = np.mean(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        # 只要该特征均值不为0（由于特征全为正数，即全部特征不为0），修改x矩阵中每个特征值，第i小时的第j个特征值 = (第i小时的第j个特征值-该特征均值)/该特征均值
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 选取80%的数据作为训练用，剩余的数据作为验证用
# 避免由于测试数据集中含有bias，造成模型计算结果的过好或者过差，从而导致模型的偏差
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# 18*9个feature，还有一个常数项
dim = 18*9 + 1
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
w = np.zeros([dim, 1])
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for i in range(iter_time):
    # 损失函数 = 根号(sum((x*w-y)^2)/471/12)
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)
    if i % 100 == 0:
        print(str(i) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)

    # adagrad是调整learning_rate的速率，随着迭代次数的增多，降低学习速率
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

np.save('weight.npy', w)

# 载入测试数据
testdata = pd.read_csv('test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

# 测试数据为每月后10天
test_x = np.empty([240, 18*9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18*i: 18*(i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j])/std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# 预测
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

