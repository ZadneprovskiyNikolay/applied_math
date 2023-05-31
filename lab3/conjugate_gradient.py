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
        # Производим поиск на прямой для нахождения оптимального шага
        alpha, new_evals, new_g_evals = line_search(f, x, diraction, fx, g)
        f_evals += new_evals
        grad_evals += new_g_evals

        # Обновляем x, вычисляем функцию и ее градиент
        x_new = x + alpha * diraction
        fx_new = f(x_new)
        f_evals += 1
        g_new = grad(x_new)
        grad_evals += 1

        # Считаем бета для обновления промежутка поиска
        if np.dot(g, g) == 0:
            beta = 0
        else:
            beta = np.dot(g_new, g_new - g) / np.dot(g, g)

        # Обновляем направление поиска
        d_new = -g_new + beta * diraction

        # Проверяем сходимость
        if abs(fx_new - fx) < tol:
            break

        # Обновляем переменные для следующей итерации
        x, g, diraction, fx = x_new, g_new, d_new, fx_new
        trajectory.append(x)

        if abs(f(trajectory[-1]) - f(trajectory[-2])) < 1e-10:
            break

    return np.array(trajectory), f_evals, grad_evals, iteration


def line_search(f, x, d, f_x, grad_f_x, alpha_init=1, c=0.4):
    f_evals = 0
    grad_evals = 0
    alpha = alpha_init
    while f(x + alpha * d) > f_x + c * alpha * np.dot(grad_f_x, d):
        f_evals += 1
        alpha /= 2

    # Возвращаем оптимальных шаг и кол-во подсчетов функции и градиента
    return alpha, f_evals, grad_evals