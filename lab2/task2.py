import numpy as np
import matplotlib.pyplot as plt
import bigfloat
import pandas as pd

df = pd.read_csv('data/2_train.csv')
x1_train, x2_train, y_train = [df[k].values for k in ['x1', 'x2', 'y']]


def get_features(x1, x2):
    matrix = []
    for k in range(len(x1)):
        array = list()
        for i in range(10):
            for j in range(10):
                if i + j <= 10:
                    array.append((x1[k] ** i) * (x2[k] ** j))
        matrix.append(np.array(array))
    return np.array(matrix)


def transform_y_for_one_vs_all(y, iter):
    for i in range(len(y)):
        if y[i] == iter:
            y[i] = 1
        else:
            y[i] = 0
    return y


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def sigmoid(x):
    n = len(x)
    ones = np.array(n * [1])
    return ones / (ones + np.array([float(bigfloat.exp(i)) for i in x]))


def h(theta, x):
    # theta_transpose = np.transpose(theta)
    z = np.matmul(x, theta)
    return sigmoid(z)


def cost_function(y, theta, x):
    ones = np.array(len(y) * [1])
    y = np.transpose(y)
    a = np.transpose(ones - h(theta, x))
    a[a == 0] = 0.01
    sum = np.matmul(-y, np.log(np.transpose(h(theta, x)))) - np.matmul((ones - y),
                                                                       np.log(a))
    return sum


def grad(theta, x, y):
    x_transpose = np.transpose(x)
    grads = (1 / len(y)) * np.matmul(x_transpose, y - np.transpose(h(theta, x)))
    return grads


def gradient_decent(theta, x, y, alpha, iteration):
    for i in range(iteration):
        print("iter" + str(i) + ': ' + str(cost_function(y, theta, x)))
        theta = theta - alpha * grad(theta, x, y)
    return theta


# def predict(theta, x1, x2):
#     return h(theta, x1, x2).argmax(axis=1)


x = get_features(x1_train, x2_train)
theta0 = np.zeros(shape=(64, 1))
theta = theta0
y_1 = np.array(transform_y_for_one_vs_all(y_train, 0))
y_1 = np.reshape(y_1, newshape=(500, 1))
alpha = 1e-20
iteration = 100
theta = gradient_decent(theta.T[0], x, y_1.T[0], alpha, iteration)
# z = np.matmul(x, theta.T[0])
# a = np.array([float(bigfloat.exp(i)) for i in z])
# print(a)

# plt.figure(figsize=(6, 6))
# plt.scatter(x1_train, x2_train, c=y_train, marker='.')
# plt.show()
#
# df = pd.read_csv('data/2_test.csv')
# x1_test, x2_test, y_test = [df[k].values for k in ['x1', 'x2', 'y']]
# print('Final accuracy: %.2f' % (accuracy(predict(theta, x1_test, x2_test), y_test) * 100) + '%')
