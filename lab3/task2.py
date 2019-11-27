import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/2.csv')
x = df.values


def pca(x, dimension):
    x_c = x - x.mean(axis=0)
    C = (x_c.T @ x_c) / (x_c.shape[0] - 1)
    L, W = np.linalg.eig(C)
    importance_of_components = L / L.sum()
    important_indexes = np.argsort(importance_of_components)[-dimension:][::-1]
    x_projected = x_c @ W[:, important_indexes]
    return x_projected


x_projected = pca(x, 2)
fig = plt.figure(figsize=(16, 16))
# ax = fig.add_subplot(111, projection='3d')
plt.scatter(x_projected.T[0], x_projected.T[1])
plt.show()
