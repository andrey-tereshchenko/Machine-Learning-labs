import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/1.csv')
x, y = df['x'].values, df['y'].values


def get_features(x):
    return np.stack([np.ones_like(x), x], axis=1)


def predict(theta, x):
    return h(theta, x) > 0.5


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


x = get_features(x)
y = np.expand_dims(y, axis=1)


def sigmoid(x):
    # TODO
    pass


def h(theta, x):
    # TODO
    return 0 * x[:, :1]


def grad(theta, x, y):
    # TODO
    pass


theta0 = np.zeros(shape=(2, 1))
theta = theta0

# TODO

x_plot = np.linspace(0, 1)
y_plot = h(theta, get_features(x_plot)).T[0]

plt.scatter(x.T[1], y.T[0], marker='.')
plt.plot(x_plot, y_plot, c='red')
plt.show()

print('Final accuracy: %.2f' % (accuracy(predict(theta, x), y) * 100) + '%')
