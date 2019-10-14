import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/4.csv')
x1, x2, y = df['x1'].values, df['x2'].values, df['y'].values


def list_x(x1, x2):
    x = [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2, x1 * x2 ** 2,
         x2 * x1 ** 2, x1 ** 3, x2 ** 3]
    return x


def h(x1, x2, theta):
    x = np.transpose(np.array(list_x(x1, x2)))
    return np.matmul(theta, x)


def cost_function(x1, x2, y, theta):
    m = len(y)
    param = (1 / (2 * m))
    sum = 0
    for i in range(len(y)):
        sum += (h(x1[i], x2[i], theta) - y[i]) ** 2
    return param * sum


def grad(theta):
    m = len(y)
    grads = []
    array_of_sum = 10 * [0]
    for i in range(m):
        difference = h(x1[i], x2[i], theta) - y[i]
        x = list_x(x1[i], x2[i])
        for j in range(len(x)):
            array_of_sum[j] += difference * x[j]
    for i in range(10):
        grads.append((1 / m) * array_of_sum[i])
    return grads


theta_initial = np.array(10 * [1])
learning_rate = 8e-4
theta = theta_initial

for _ in range(300):
    print("J:" + str(cost_function(x1, x2, y, theta)))
    grads = np.array(grad(theta))
    theta = theta - learning_rate * grads


def points_for_graph(theta):
    x1 = []
    x2 = []
    y = []
    for i in np.arange(-5, 5, 0.2):
        for j in np.arange(-3, 3, 0.2):
            x1.append(i)
            x2.append(j)
            y.append(h(i, j, theta))
    return x1, x2, y


print("Theta: " + str(theta))

x_1, x_2, y_1 = points_for_graph(theta)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y)
ax.plot_trisurf(x_1, x_2, y_1, color='red')
plt.show()
