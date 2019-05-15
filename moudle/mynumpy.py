import numpy as np

arr = np.array([1, 1, 1, 134, 45, 3, 46, 45, 65, 3, 23424, 234, 12, 12, 3, 546, 1, 2])

print(np.where(arr==3))

x = np.array([[1,2,3],
        [4,5,6],
        [7,8,9]])

print(np.delete(x, [0,2], 0))
# print(np.delete(x, 0, 0))
x_norm = np.linalg.norm(x, axis=1, ord=1, keepdims=True)
print(x_norm)

index_matrix = np.argsort(-x, axis=1)
print(index_matrix)

a = np.random.randn(100)

b = np.sum(x, axis=0)
print(b)