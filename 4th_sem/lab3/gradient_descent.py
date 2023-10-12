import numpy as np

def gradient_descent(f, grad, start_point, step, iterations):
    x = start_point
    trajectory = [x]
    for iteration in range(1, iterations+1):
        x = x - grad(x)*step
        trajectory.append(x)
        if abs(f(trajectory[-1]) - f(trajectory[-2])) < 1e-10:
            break
    
    f_evals = 0
    grad_evals = iterations

    return np.array(trajectory), f_evals, grad_evals, iteration

# https://en.wikipedia.org/wiki/Wolfe_conditions  
# http://www.machinelearning.ru/wiki/images/6/6b/MO17_practice1.pdf
# http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%81%D0%BF%D1%83%D1%81%D0%BA%D0%B0
def gradient_armijo_descent(f, grad, start_point, step, iterations):
    x = start_point
    c = 0.3
    trajectory = [x]
    f_evals = 0
    grad_evals = iterations

    def find_armijo_step():
        nonlocal f_evals, grad_evals, step
        while f(x - grad(x)*step) > f(x) - c*step*np.linalg.norm(grad(x))**2:
            step = step / 2
            f_evals += 2
            grad_evals += 2
        return step
    
    for iteration in range(1, iterations+1):
        step = find_armijo_step()
        x = x - grad(x)*step
        grad_evals += 1
        trajectory.append(x)

    return np.array(trajectory), f_evals, grad_evals, iteration

