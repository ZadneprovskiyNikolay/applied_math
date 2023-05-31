import numpy as np

def seidel_solve(A, b, eps=1e-4, iters=1e5):
    n = len(A)
    x = np.zeros(n)
    for _ in range(int(iters)):
        x_new = np.copy(x)
        for i in range(n):
            # Calculate the two summation terms in the iterative formula
            s1 = sum(A[i][j] * x_new[j] for j in range(i))  # Summation of A[i][j] * x_new[j] for j < i
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # Summation of A[i][j] * x[j] for j > i
            x_new[i] = (b[i] - s1 - s2) / A[i][i]  # Update the value of x[i] using the iterative formula

        # Check for convergence by calculating the Euclidean norm of the difference between x_new and x
        if np.linalg.norm(x_new - x) <= eps:
            return x
        
        x = x_new  # Update the solution vector    

    return x