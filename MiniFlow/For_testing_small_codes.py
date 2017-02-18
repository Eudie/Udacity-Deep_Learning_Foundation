import numpy as np

X = np.array([[-1., -2.], [-1, -2]])
W = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

a = np.dot(X, W) + b
print(X)
print(W)
print(a)
