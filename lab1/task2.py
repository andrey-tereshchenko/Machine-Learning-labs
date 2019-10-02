import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/2.csv')
x, y = df['x'].values, df['y'].values


def h(theta, x):
    return theta[0] + theta[1] * x + theta[2] * x ** 2 + theta[3] * x ** 3


def difference_array(array_1, array_2, learning_rate):
    difference = []
    if len(array_1) == len(array_2):
        for i in range(len(array_1)):
            difference.append(array_1[i] - learning_rate * array_2[i])
    return difference


def cost_function(theta):
    m = len(x)
    param = (1 / (2 * m))
    sum = 0
    for i in range(len(x)):
        sum += (h(theta, x[i]) - y[i]) ** 2
    return param * sum


def grad(theta):
    m = len(y)
    grads = []
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    for i in range(m):
        difference = h(theta, x[i]) - y[i]
        sum_1 += difference
        sum_2 += difference * x[i]
        sum_3 += difference * x[i] ** 2
        sum_4 += difference * x[i] ** 3
    grads.append((1 / m) * sum_1)
    grads.append((1 / m) * sum_2)
    grads.append((1 / m) * sum_3)
    grads.append((1 / m) * sum_4)
    return grads


theta_initial = [1, 1, 1, 1]
learning_rate = 3e-3
theta = theta_initial

for _ in range(400):
    print("J:" + str(cost_function(theta)))
    grads = grad(theta)
    theta = difference_array(theta, grads, learning_rate)

print('theta: ' + str(theta))


def points_for_graph(theta):
    x = [i for i in np.arange(-5, 5, 0.1)]
    y = [h(theta, x[i]) for i in range(len(x))]
    return x, y

x1, y1 = points_for_graph(theta)
plt.scatter(x, y, marker='.')
plt.plot(x1, y1, c='red')
plt.show()
