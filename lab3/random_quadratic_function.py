import numpy as np

def random_quadratic_function(n, k):
    # generate a random orthogonal matrix Q of size n x n
    # which preserves dot products, i.e., Q @ Q^T = Q^T @ Q = I
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # generate a diagonal matrix D of size n x n 
    # with increasing eigenvalues in the range [1, k]
    D = np.diag(np.linspace(1, k, n))
    
    # form a positive definite matrix A of size n x n
     #Thus, the product of Q, D and the transposed Q gives a random positive definite matrix of size nxn with a conditionality number k.
    A = Q @ D @ Q.T

    # generate a random vector b of size n
    b = np.random.randn(n)

    # define the quadratic function f and its gradient
    def f(x):
        return 0.5 * x.T @ A @ x - b.T @ x

    def grad_f(x):
        return A @ x - b

    return f, grad_f