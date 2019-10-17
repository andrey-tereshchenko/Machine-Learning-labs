import numpy as np

matrix = np.array([[1, 2, 3, 5],
                   [2, 3, 4, 1],
                   [4, 1, 2, 2]])
max = np.argmax(matrix, axis=0)
print(max)
