import numpy as np

def conjugate_gradient_descent(f, grad, point, tol=1e-6, iterations=1000, **kwargs):
    f_evals = 0
    grad_evals = 0
    x = point
    trajectory = [x]

    g = grad(x)
    grad_evals += 1
    fx = f(x)
    f_evals += 1

    diraction = -g
    for iteration in range(1, iterations+1):
        # Perform line search to find optimal step size
        alpha, new_evals, new_g_evals = line_search(f, x, diraction, fx, g)
        f_evals += new_evals
        grad_evals += new_g_evals

        # Update x and evaluate function and gradient at new point
        x_new = x + alpha * diraction
        fx_new = f(x_new)
        f_evals += 1
        g_new = grad(x_new)
        grad_evals += 1

        # Compute beta for updating search direction
        if np.dot(g, g) == 0:
            beta = 0
        else:
            beta = np.dot(g_new, g_new - g) / np.dot(g, g)

        # Update search direction
        d_new = -g_new + beta * diraction

        # Check for convergence
        if abs(fx_new - fx) < tol:
            break

        # Update variables for next iteration
        x, g, diraction, fx = x_new, g_new, d_new, fx_new
        trajectory.append(x)

        if abs(f(trajectory[-1]) - f(trajectory[-2])) < 1e-10:
            break

    # Return trajectory, function and gradient evaluations
    return np.array(trajectory), f_evals, grad_evals, iteration


def line_search(f, x, d, f_x, grad_f_x, alpha_init=1, c=0.4):
    # Initialize counters for function and gradient evaluations
    g_evals = 0
    evals = 0

    # Initialize step size and evaluate function and gradient at current point
    alpha = alpha_init

    # Backtracking line search
    while f(x + alpha * d) > f_x + c * alpha * np.dot(grad_f_x, d):
        # Update step size and evaluate function at new point
        evals += 1
        alpha /= 2

    # Return optimal step size and function/gradient evaluations
    return alpha, evals, g_evals