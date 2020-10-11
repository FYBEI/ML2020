import numpy as np
from hw2 import utils

np.random.seed(0)
X_train_path = 'data/X_train'
Y_train_path = 'data/Y_train'
X_test_path = 'data/X_test'
output_path = 'data/output_{}.csv'

# parse csv to numpy array
with open(X_train_path) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_path) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_path) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

# Normalize training and testing data
X_train, X_mean, X_std = utils.normalize(X_train, train=True)
X_test, _, _ = utils.normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = utils.train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('train_size:', train_size)
print('dev_size:', dev_size)
print('test_size:', test_size)
print('data_dim:', data_dim)


# 初始化权重值w和偏差值b，初始值为0
w = np.zeros((data_dim,))
b = np.zeros((1,))

# 训练的迭代次数，批次大小，训练效率
max_iter = 10
batch_size = 8
learning_rate = 0.2

# 保存每次迭代的损失值和精确度
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# 计算参数更新的次数
step = 1

# 迭代训练，每次迭代中包含多次小批次迭代
for epoch in range(max_iter):
    print(step)
    X_train, Y_train = utils.shuffle(X_train, Y_train)

    # 分为小批次迭代
    for idx in range(int(np.floor(train_size/batch_size))):
        X = X_train[idx*batch_size: (idx+1)*batch_size]
        Y = Y_train[idx*batch_size: (idx+1)*batch_size]

        # 计算梯度
        w_grad, b_grad = utils.gradient(X, Y, w, b)

        # 梯度递减，随着步数增加降低更新速率
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1

    # 保存每次迭代的损失值和精确度
    y_train_pred = utils.f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    acc = utils.accuracy(Y_train_pred, Y_train)
    print('train acc:', acc)
    train_acc.append(acc)
    loss = utils.cross_entropy_loss(y_train_pred, Y_train) / train_size
    print('train loss:', loss)
    train_loss.append(loss)

    y_dev_pred = utils.f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    acc = utils.accuracy(Y_dev_pred, Y_dev)
    dev_acc.append(acc)
    loss = utils.cross_entropy_loss(y_dev_pred, Y_dev) / dev_size
    dev_loss.append(loss)

# 绘制模型在训练模型和强化模型上的迭代变化
import matplotlib.pyplot as plt

# 绘制损失函数变化图像
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# 绘制精确度变化图像
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# 预测测试数据，并写入csv
predictions = utils.predict(X_test, w, b)
with open(output_path.format('logistic'), 'w') as f:
    f.write('id, label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

ind = np.argsort(np.abs(w))[::-1]
with open(X_test_path) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
