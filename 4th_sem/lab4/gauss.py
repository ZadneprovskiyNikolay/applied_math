import numpy as np

def gaussian_solve(A, b):
    n = len(b)

    # Gaussian elimination with partial pivoting
    for i in range(n):
        # Find the maximum element in the current column
        max_elem = abs(A[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > max_elem:
                max_elem = abs(A[k][i])
                max_row = k

        # Swap rows to bring the maximum element to the diagonal position
        for k in range(i, n):
            tmp = A[max_row][k]
            A[max_row][k] = A[i][k]
            A[i][k] = tmp
        tmp = b[max_row]
        b[max_row] = b[i]
        b[i] = tmp

        # Perform row operations to eliminate variables below the diagonal
        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]
            b[k] += c * b[i]

    # Back substitution to solve for x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x