import numpy as np

def lu_solve(A, b):
    # Create a matrix for storing the LU decomposition
    lu_matrix = np.matrix(np.zeros([A.shape[0], A.shape[1]]))
    n = A.shape[0]

    # LU decomposition
    for k in range(n):
        # Calculate all residual elements in the k-th row
        for j in range(k, n):
            lu_matrix[k, j] = A[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        # Calculate all residual elements in the k-th column
        for i in range(k + 1, n):
            if lu_matrix[k, k] == 0:
                lu_matrix[i, k] = 0
            else:    
                lu_matrix[i, k] = (A[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

    # Forward substitution to solve for y
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i] - lu_matrix[i, :i] * y[:i]

    # Back substitution to solve for x
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0]) / lu_matrix[-i, -i]

    return np.ravel(x)