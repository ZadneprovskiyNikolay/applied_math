def dichotomy_minimize(f, search_range, E):
    f_evals = 0
    l, r = search_range
    delta = E / 2
    x_min = None
    y_min = None
    while r - l > E:
        m1 = (l + r - delta) / 2 
        m2 = (l + r + delta) / 2 
        f1, f2 = f(m1), f(m2)
        f_evals += 2

        if f1 > f2:
            y_min = f2
            x_min = m2
            l = m1
        else:
            y_min = f1
            x_min = m1
            r = m2
        
    return x_min, y_min, f_evals