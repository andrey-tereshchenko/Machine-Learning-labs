import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/3.csv')
x, y = df['x'].values, df['y'].values


def h(k, b, x):
    return k * x + b


def J(k, b):
    m = len(x)
    param = (1 / (2 * m))
    sum = 0
    for i in range(len(x)):
        sum += (h(k, b, x[i]) - y[i]) ** 2
    return param * sum


def grad(k, b):
    m = len(y)
    sum_1 = 0
    sum_2 = 0
    for i in range(m):
        sum_2 += h(k, b, x[i]) - y[i]
        sum_1 += (h(k, b, x[i]) - y[i]) * x[i]
    grad_k = (1 / m) * sum_1
    grad_b = (1 / m) * sum_2
    return grad_k, grad_b


k0, b0 = 0, 0
learning_rate = 0.9

k, b = k0, b0
for _ in range(100):
    # print("k:" + str(k) + " b:" + str(b))
    print("J:" + str(J(k, b)))
    grad_k, grad_b = grad(k, b)
    k, b = k - learning_rate * grad_k, b - learning_rate * grad_b

print("k: " + str(k) + '\nb: ' + str(b))
plt.scatter(x, y, marker='.')
plt.plot([0, 1], [b, k + b], c='red')
plt.show()
