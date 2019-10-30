import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/2_train.csv')
x1_train, x2_train, y_train = [df[k].values for k in ['x1', 'x2', 'y']]


def get_features(x1, x2):
    matrix = []
    for k in range(len(x1)):
        array = list()
        for i in range(11):
            for j in range(11):
                if i + j <= 10:
                    array.append((x1[k] ** i) * (x2[k] ** j))
        matrix.append(np.array(array))
    return np.array(matrix)


def normalize(x):
    max_in_column = np.max(x, axis=0)
    min_in_column = np.min(x, axis=0)
    difference = max_in_column - min_in_column
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if difference[j] != 0:
                x[i][j] = (x[i][j] - min_in_column[j]) / difference[j]
    return x


def transform_y_for_one_vs_all(y, iter):
    transform_y = []
    for i in range(len(y)):
        if y[i] == iter:
            transform_y.append(1)
        else:
            transform_y.append(0)
    return transform_y


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def sigmoid(x):
    n = len(x)
    ones = np.array(n * [1])
    return ones / (ones + np.exp(x))


def h(theta, x):
    z = np.matmul(x, theta)
    return sigmoid(z)


def cost_function(y, theta, x):
    ones = np.array(len(y) * [1])
    y = np.transpose(y)
    b = np.transpose(h(theta, x))
    a = np.transpose(ones - h(theta, x))
    sum = np.matmul(-y, np.log(b)) - np.matmul((ones - y),
                                               np.log(a))
    return sum


def grad(theta, x, y):
    x_transpose = np.transpose(x)
    grads = (1 / len(y)) * np.matmul(x_transpose, y - np.transpose(h(theta, x)))
    return grads


def gradient_decent(theta, x, y, alpha, iteration):
    for i in range(iteration):
        # print("iter" + str(i) + ': ' + str(cost_function(y, theta, x)))
        theta = theta - alpha * grad(theta, x, y)
    return theta


def get_coordinate_for_plot(x1, x2, y):
    x1_plot = []
    x2_plot = []
    for i in range(len(y)):
        if y[i] == 1:
            x1_plot.append(x1[i])
            x2_plot.append(x2[i])
    return x1_plot, x2_plot


def one_vs_all(theta_0, theta_1, theta_2, x):
    h_matrix = np.array([h(theta_0, x), h(theta_1, x), h(theta_2, x)])
    y_predict = np.argmax(h_matrix, axis=0)
    return y_predict


x = get_features(x1_train, x2_train)
x = normalize(x)
theta = np.zeros(shape=(66, 1))
y_0 = np.reshape(np.array(transform_y_for_one_vs_all(y_train, 0)), newshape=(500, 1))
y_1 = np.reshape(np.array(transform_y_for_one_vs_all(y_train, 1)), newshape=(500, 1))
y_2 = np.reshape(np.array(transform_y_for_one_vs_all(y_train, 2)), newshape=(500, 1))
alpha = 30
iteration = 5000
theta_0 = gradient_decent(theta.T[0], x, y_0.T[0], alpha, iteration)
theta_1 = gradient_decent(theta.T[0], x, y_1.T[0], alpha, iteration)
theta_2 = gradient_decent(theta.T[0], x, y_2.T[0], alpha, iteration)

df = pd.read_csv('data/2_test.csv')
x1_test, x2_test, y_test = [df[k].values for k in ['x1', 'x2', 'y']]
x_test = get_features(x1_test, x2_test)
x_test = normalize(x_test)
train_result = one_vs_all(theta_0, theta_1, theta_2, x)
test_result = one_vs_all(theta_0, theta_1, theta_2, x_test)

# x1_plot, x2_plot = get_coordinate_for_plot(x1_train, x2_train, result)
# plt.figure(figsize=(6, 6))
# plt.scatter(x1_train, x2_train, c=y_train, marker='.')
# plt.plot(x1_plot, x2_plot, c='red')
# plt.show()
print('Train accuracy: %.2f' % (accuracy(train_result, y_train) * 100) + '%')
print('Final accuracy: %.2f' % (accuracy(test_result, y_test) * 100) + '%')
