import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

train_data = np.genfromtxt('data/2_train.csv', delimiter=',', skip_header=1)

X = PolynomialFeatures(degree=10, include_bias=False).fit_transform(train_data[:, :-1])
x_train = np.hstack((np.ones((train_data.shape[0], 1)), X))
y_train = train_data[:, -1:].ravel()

test_data = np.genfromtxt('data/2_test.csv', delimiter=',', skip_header=1)

X = PolynomialFeatures(degree=10, include_bias=False).fit_transform(test_data[:, :-1])
x_test = np.hstack((np.ones((test_data.shape[0], 1)), X))
y_test = test_data[:, -1:].ravel()

df = pd.read_csv('data/2_test.csv')
x1_test, x2_test, y_test = [df[k].values for k in ['x1', 'x2', 'y']]


def transform_y_for_classes(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def h(w, b, x):
    return np.stack(
        [(np.matmul(x, w.T[0]) - b[0]),
         (np.matmul(x, w.T[1]) - b[1]),
         (np.matmul(x, w.T[2]) - b[2])], axis=1)


def predict(w, b, x):
    return h(w, b, x).argmax(axis=1)


def H(x):
    return np.sign(x) / 2 + 1 / 2


def grad(w, b, x, y, c):
    h1 = h(w, b, x)

    dw1 = -c * ((H(1 - y.T[0] * h1.T[0]) * y.T[0])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[0]
    dw2 = -c * ((H(1 - y.T[1] * h1.T[1]) * y.T[1])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[1]
    dw3 = -c * ((H(1 - y.T[2] * h1.T[2]) * y.T[2])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[2]
    b = c * (H(1 - y * h1) * y).sum(axis=0)

    return (np.stack([dw1, dw2, dw3], axis=1), b)


w = np.zeros(shape=(66, 3))
b = [0, 0, 0]

c = 0.0001
lr = 0.0000001
encoded_y_train = transform_y_for_classes(y_train)
for i in range(encoded_y_train.T[0].size):
    for j in range(encoded_y_train[0].size):
        if encoded_y_train[i][j] == 0:
            encoded_y_train[i][j] = -1

for i in range(200000):
    dw, db = grad(w, b, x_train, encoded_y_train, c)
    w = w - lr * dw
    b = b - lr * db

plt.figure(figsize=(6, 6))
plt.scatter(x1_test, x2_test, c=predict(w, b, x_test), marker='.')
plt.show()

print('Final accuracy: %.2f' % (accuracy(predict(w, b, x_test), y_test) * 100) + '%')
print(h(w, b, x_test))
