import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/2_train.csv')
x1_train, x2_train, y_train = [df[k].values for k in ['x1', 'x2', 'y']]


def get_features(x1, x2):
    matrix = []
    for i in range(len(x1)):
        matrix.append(np.array([x1[i], x2[i]]))
    return np.array(matrix)


def normalize(l1, l2, x, sigma):
    res1 = l1 - x
    res2 = l2 - x
    degrees = []
    for i in range(res1.shape[0]):
        d = []
        d.append(-norma_in_two_degree(res1[i]) / (2 * sigma ** 2))
        d.append(-norma_in_two_degree(res2[i]) / (2 * sigma ** 2))
        degrees.append(d)
    degrees = np.array(degrees).reshape(500, 2)
    new_x = np.exp(degrees)
    # print(new_x)
    # print(new_x.shape)
    return new_x


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def transform_y_for_one_vs_all(y, iter):
    transform_y = []
    for i in range(len(y)):
        if y[i] == iter:
            transform_y.append(1)
        else:
            transform_y.append(-1)
    return transform_y


def norma_in_two_degree(w):
    s = 0
    for i in w:
        # s += i[0] ** 2
        s += i ** 2
    return s


def h(x, w, b):
    return np.matmul(x, w.T) - b


def cost_function(x, y, w, b, C):
    sum_vector = 1 - y * h(x, w, b).T[0]
    sum_vector[sum_vector < 0] = 0
    sum = np.sum(sum_vector)
    norma = norma_in_two_degree(w[0])
    sum = C * sum + norma
    return sum


def grads(x, y, w, b, C):
    sum_vector = 1 - y * h(x, w, b).T[0]
    sum_vector[sum_vector < 0] = 0
    sum_vector[sum_vector > 0] = 1
    result = sum_vector * y
    sum = np.matmul(result.T, x)
    grads_w = -C * sum + 2 * w
    grad_b = C * np.matmul(sum_vector, y.T)
    return grads_w, grad_b


def gradient_decent(x, y, w, b, C, alpha, alpha2, iteration):
    w = w.T
    for i in range(iteration):
        print("iter" + str(i) + ': ' + str(cost_function(x, y, w, b, C)))
        grad_w, grad_b = grads(x, y, w, b, C)
        # print(grad_w)
        # print(grad_b)
        w = w - alpha * grad_w
        b = b + alpha2 * grad_b
        # print(w,b)
    return w, b


def get_coordinate_for_plot(x1, x2, y):
    x1_plot = []
    x2_plot = []
    for i in range(y.shape[1]):
        if y[0][i] == 1:
            x1_plot.append(x1[i])
            x2_plot.append(x2[i])
    return x1_plot, x2_plot


def one_vs_all(w_0, b_0, w_1, b_1, w_2, b_2, x):
    h_matrix = np.array([h(x, w_0, b_0).reshape(500, ), h(x, w_1, b_1).reshape(500, ),
                         h(x, w_2, b_2).reshape(500, )])
    y_predict = np.argmax(h_matrix, axis=0)
    # print(y_predict.shape)
    # print(y_predict)
    return y_predict


x = get_features(x1_train, x2_train)
x = normalize(x[50], x[43], x, 13)
y_0 = np.array(transform_y_for_one_vs_all(y_train, 2))
# y_0 = np.array(transform_y_for_one_vs_all(y_train, 2)).reshape(1, 500)
# y_1 = np.array(transform_y_for_one_vs_all(y_train, 1)).reshape(1, 500)
# y_2 = np.array(transform_y_for_one_vs_all(y_train, 2)).reshape(1, 500)
# w = np.array([0, 1]).reshape(2,1)
w = np.array([-1000, 1000]).reshape(2, 1)
b = 0
C = 1e-2
alpha = 1e-4
alpha2 = 1e-3
iteration = 1000
# print(h(x, w, b).T[0])
# print(y_train.shape)
w_0, b_0 = gradient_decent(x, y_0, w, b, C, alpha, alpha2, iteration)
w_0 = w_0[0]
print(w_0, b_0)
# w_1, b_1 = gradient_decent(x, y_1, w, b, C, alpha, iteration)
# w_2, b_2 = gradient_decent(x, y_2, w, b, C, alpha, iteration)
# print(h(x, w_0, b_0)[0])
# print(h(x, w_1, b_1)[0])
# print(h(x, w_2, b_2)[0])
# train_result = one_vs_all(w_0, b_0, w_1, b_1, w_2, b_2, x)
# print(train_result)
# print(y_train)
# print(w.shape)
train_result = h(x, w_0, b_0).reshape(1, 500)
# print(train_result)
# train_result = h(x, w, b).reshape(1, 500)
# print("TRAIN RESULT:" + str(train_result))
train_result[train_result < 0] = -1
train_result[train_result >= 0] = 1
x1_plot, x2_plot = get_coordinate_for_plot(x.T[0], x.T[1], train_result)
plt.figure(figsize=(6, 6))
plt.scatter(x.T[0], x.T[1], c=y_train, marker='.')
# b_0 = -0.1
# w_0[0] = 5
# w_0[1] = -4
# plt.scatter(h(x,w,b), [0] * x.shape[0], c='black', marker='x')
plt.plot([0, 1], [b_0 / w_0[1], -w_0[1] / w_0[0] + b_0 / w_0[1]], c='green')
# plt.plot(x1_plot, x2_plot, c='red')
plt.show()
print('Train accuracy: %.2f' % (accuracy(train_result, y_0) * 100) + '%')
