import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/1.csv')
x, y = df['x'].values, df['y'].values


def h(k, b):
    # TODO
    return np.zeros_like(x)


def J(k, b):
    # TODO
    return 0


def grad(k, b):
    # TODO
    return 0, 0


k0, b0 = 0, 0
learning_rate = 1e-1

k, b = k0, b0
for _ in range(100):
    # Update k, b
    pass

plt.scatter(x, y, marker='.')
plt.plot([0, 1], [b, k + b], c='red')
plt.show()
