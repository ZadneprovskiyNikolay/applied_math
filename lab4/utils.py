import numpy as np

def generate_random_matrix(k, n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = np.random.choice([-1, -2, -3, -4])

    for i in range(n):
        if i == 0:
            A[i, i] = -np.sum(A[i]) + 10 ** (-k)
        else:
            A[i, i] = -np.sum(A[i])
    return A

def generate_hilbert_matrix(n):
    A = np.zeros((n, n))
    for i in range(1, n+1):
        for j in range(1, n+1):
            A[i-1][j-1] = 1.0 / (i + j - 1)
    return A

def generate_b_vector(n):
    return np.arange(1, n + 1)