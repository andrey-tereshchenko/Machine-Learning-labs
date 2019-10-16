import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/1.csv')
x, y = df['x'].values, df['y'].values
N = len(x)


def get_features(x):
    return np.stack([np.ones_like(x), x], axis=1)


def predict(theta, x):
    return h(theta, x) > 0.5


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def sigmoid(x):
    n = len(x)
    ones = np.array(n * [1])
    return ones / (ones + np.exp(x))


def h(theta, x):
    # theta_transpose = np.transpose(theta)
    z = np.matmul(x, theta)
    return sigmoid(z)


def cost_function(y, theta, x):
    ones = np.array(N * [1])
    y = np.transpose(y)
    sum = np.matmul(-y, np.log(np.transpose(h(theta, x)))) - np.matmul((ones - y),
                                                                       np.log(np.transpose(ones - h(theta, x))))
    return sum


def grad(theta, x, y):
    x_transpose = np.transpose(x)
    grads = (1 / N) * np.matmul(x_transpose, y - np.transpose(h(theta, x)))
    return grads


def gradient_decent(theta, x, y, alpha, iteration):
    for i in range(iteration):
        print("iter" + str(i) + ': ' + str(cost_function(y, theta, x)))
        theta = theta - alpha * grad(theta, x, y)
    return theta


x = get_features(x)
y = np.expand_dims(y, axis=1)
theta0 = np.zeros(shape=(2, 1))
theta = theta0
alpha = 2
iteration = 100
theta = gradient_decent(theta.T[0], x, y.T[0], alpha, iteration)
x_plot = np.linspace(0, 1)
y_plot = h(theta, get_features(x_plot))

plt.scatter(x.T[1], y.T[0], marker='.')
plt.plot(x_plot, y_plot, c='red')
plt.show()

print('Final accuracy: %.2f' % (accuracy(predict(theta, x), y.T[0]) * 100) + '%')
