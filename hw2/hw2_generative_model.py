import numpy as np
from hw2 import utils

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

# 计算平均
# 获得y=0类别的X训练数据和y=1类别的X训练数据
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

# 分别计算训练数据的平均
mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# 计算covariance
data_dim = X_train_0.shape[1]
cov_0 = np.zeros((data_dim, data_dim))
data_dim = X_train_1.shape[1]
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# 共同cov按照class1和class2所占比例作为权重得出
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_1.shape[0] + X_train_0.shape[0])

# 求协方差矩阵的逆
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# 直接计算权重和偏差
w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(X_train_0.shape[0])) / X_train_1.shape[0]

Y_train_pred = 1 - utils.predict(X_train, w, b)
print('Training accuracy: {}'.format(utils.accuracy(Y_train_pred, Y_train)))

# Predict testing labels
predictions = 1 - utils.predict(X_test, w, b)
with open(output_path.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_path) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])


