import numpy as np

matrix = np.array([[1, 2, 3, 5],
                   [2, 3, 4, 1],
                   [4, 1, 2, 2]])

min = np.min(matrix, axis=0)
max = np.max(matrix, axis=0)
difference = max - min
print(max)
print(min)
print(difference)
