import pandas as pd
import numpy as np
import csv

train_data = pd.read_csv('train.csv', encoding='big5')
train_data = train_data.iloc[:, 3:]
train_data[train_data == 'NR'] = 0

# 4320*24
train_data = train_data.to_numpy()

# 转化为18*(24*20*12), 即行为18个features，列为每个小时的features的值
day = 20*12
col_data = np.zeros([18, 24*20*12], dtype=float)
for i in range(day):
    col_data[:, i*24: (i+1)*24] = train_data[i*18: (i+1)*18, :]

# 转化为(18*9)*576，即每9个小时的18个features全部作为特征，第10小时的PM2.5作为目标
x = np.zeros([18*9, 576], dtype=float)
y = np.zeros([1, 576], dtype=float)
row = 1
col = 1
i = 0
for j in range(24*20*12):
    if (j+1)%10 == 0:
        y[0, i] = col_data[9, j]
        i += 1
    else:
        x[(row-1)*18: row*18, col-1] = col_data[:, j]
        if row == 9:
            col += 1
            row = 1
        else:
            row += 1
x = np.concatenate((x, np.ones([1, 576], dtype=float)), axis=0)

data = np.random.random([1, 18*9 + 1])
rate = 100
adagrad = np.zeros([1, 18*9 + 1])
eps = 0.0000000001
for i in range(200001):
    loss = np.sqrt(np.sum((data@x - y)**2)/163)
    gradient = 2*np.dot(np.dot(data, x) - y, x.transpose())
    adagrad += gradient ** 2
    data = data - rate * gradient / np.sqrt(adagrad + eps)
    if i%5000 == 0:
        print(data)
        print(i, loss)

np.save('data.npy', data)

# 载入测试数据
testdata = pd.read_csv('test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

# 测试数据为每月后10天
test_x = np.empty([18*9, 240], dtype=float)
for i in range(240):
    for j in range(9):
        test_x[j*18: (j+1)*18, i] = test_data[i*18: (i+1)*18, j]
test_x = np.concatenate((test_x, np.ones([1, 240], dtype=float)), axis=0)

# 预测
data = np.load('data.npy')
ans_y = np.dot(data, test_x)

with open('submit2.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[0][i]]
        csv_writer.writerow(row)
