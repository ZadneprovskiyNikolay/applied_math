import numpy as np
from one_dim_optimization import dichotomy_minimize

# http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%81%D0%BF%D1%83%D1%81%D0%BA%D0%B0
def fastest_descent(f, grad, start_point, step, iterations):
    x = start_point
    trajectory = [x]
    f_evals = 0
    grad_evals = 0

    for iteration in range(1, iterations+1):
        diraction = -grad(x)
        grad_evals += 1
        x_start = x
        f_value_start = f(x_start)
        f_evals += 1
        # step in direction of -grad until find x >= x_start
        x = x + diraction*step
        while f(x) < f_value_start:
            f_evals += 1
            x = x + diraction*step
        x_end = x
        
        # use 1 dimensional optimization for f(step_from_start) to find minimum in range(step_start, step_end)
        step_start = 0
        step_end = np.linalg.norm(x_end - x_start) / np.linalg.norm(diraction)
        f_step = lambda step: f(x_start + diraction*step)
        E = min(0.01, abs(f_step(step_end) - f_step(step_start)))
        if E < 1e-10:
            break
        f_evals += 2
        step_needed, _, opt_f_evals = dichotomy_minimize(f_step, (step_start, step_end), E)
        f_evals += opt_f_evals
        x = x_start + diraction*step_needed
        trajectory.append(x)

    return trajectory, f_evals, grad_evals, iteration