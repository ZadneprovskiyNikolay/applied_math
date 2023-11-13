# %%
import json
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", help="input file")
parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction)


args = parser.parse_args()

if args.file is None:
    args.file = 'lin_system_min.json'
    args.verbose = True

n = 0 # Кол-во переменных

def parse_input(file_name):
    global n

    with open(file_name, 'r') as f:
        data = json.load(f)

    if any(key not in data for key in ['f', 'goal', 'constraints']):
        raise Exception('incorrect input file structure')

    constraints = data['constraints']
    M = len(constraints) # Кол-во уравнений/неравенств
    n = len(constraints[0]['coefs'])
    A = [[0] * n for _ in range(M)]
    b = []

    # Сразу приводим к канонической форме
    for constraint_idx, constraint in enumerate(constraints):        
        b.append(constraint['b'])
        A[constraint_idx][0:n] = constraint['coefs']
        if constraint['type'] != 'eq':
            new_x_coef = 1 if constraint['type'] == 'lte' else -1
            for i in range(len(A)):
                A[i].append(0 if i != constraint_idx else new_x_coef)

    c = data['f']
    c.extend([0] * (len(A[0]) - n))

    return A, b, c

A, b, c = parse_input(args.file)

if args.verbose:
    print(f'A:\n{np.array(A)}')
    print(f'b:\n{b}')
    print(f'c:\n{c}\n')

# %%
def to_tableau(A, b, c):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return np.array(xb + [z])

def tableau_to_df(tableau):
    global n
    columns = [f'x{x}' for x in range(1, n+1)] + [f's{s}' for s in range(1, len(tableau[0])-n)] + ['b']
    df = pd.DataFrame(tableau, columns=columns)
    return df

tableau = to_tableau(A, b, c)
tableau_to_df(tableau)

# %%
N = len(tableau[0])-1 # Кол-во переменных в канонической форме

# %%
import math

def can_be_improved(tableau):
    # Если любой из коэффициентов в целевой функции положительный, то существует направление роста целевой функции.
    z = tableau[-1][:-1]
    return any(x > 0 for x in z)

def get_pivot_pos(tableau, basic_var_idxs):
    z = tableau[-1][:-1]
    # Выбираем переменную с самым большим коэффициентом(производной) в целевой функции для внесения в базис
    max_z_coef_col = int(np.argmax(z))
 
    # Находим переменную которая будет удалена из базиса, для этого найдем самое ближайшее ограничение в выбранном направлении
    max_ratio = -1
    max_ratio_row = -1
    removed_basic_var_idx = -1
    for row, eq in enumerate(tableau[:-1]):
        const = eq[-1]
        if eq[max_z_coef_col] == 0 or const == 0:
            continue

        ratio = eq[max_z_coef_col] / const
        if ratio > 0 and ratio > max_ratio:
            max_ratio = ratio
            max_ratio_row = row

            for col, val in enumerate(eq[:-1]):
                if val != 0 and col != max_z_coef_col and col in basic_var_idxs:
                    removed_basic_var_idx = col
        
    return (max_ratio_row, max_z_coef_col), removed_basic_var_idx

# %%
def pivot_step(tableau, pivot_pos, basic_var_idxs, removed_basic_var_idx):
    new_tableau = np.zeros(tableau.shape)

    i, j = pivot_pos
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value # Делаем опорный элемент еденицей
    # if args.verbose:
    #     print(tableau_to_df(new_tableau), end='\n\n')

    for eq_idx, eq in enumerate(tableau):
        if eq_idx != i: # Обнуляем все остальные элементы в опорной колонке        
            multiplier = np.array(new_tableau[i]) * eq[j]
            new_tableau[eq_idx] = np.array(eq) - multiplier
        # if args.verbose:
        #     print(tableau_to_df(new_tableau), end='\n\n')

    basic_var_idxs.add(j)
    basic_var_idxs.remove(removed_basic_var_idx)

    return new_tableau

# %%
def simplex(A, b, c):
    global n, N

    tableau = to_tableau(A, b, c)

    basic_var_idxs = set(range(N-n, N))

    if args.verbose:
        print(f'start tableau:\n{tableau_to_df(tableau)}\n\n')
    while can_be_improved(tableau):
        pivot_pos, removed_basic_var_idx = get_pivot_pos(tableau, basic_var_idxs)
        tableau = pivot_step(tableau, pivot_pos, basic_var_idxs, removed_basic_var_idx)
        if args.verbose:
            print(f'pivot = {pivot_pos}, step:\n{tableau_to_df(tableau)}\n\n')
    
    if args.verbose:
        print(f'end tableau:\n{tableau_to_df(tableau)}\n\n')

    solution = get_solution(tableau)
    if args.verbose:
        print(f'solution: {solution}')
    return solution

def get_solution(tableau):
    columns = np.array(tableau).T
    solution = []
    for column in columns[:-1]:
        val = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            val = columns[-1][one_index]
        solution.append(val)

    # # Выражаем изначальные переменные
    tableau = to_tableau(A, b, c)
    for i in range(n):
        eq = tableau[i]
        div = eq[i]
        eq[i] = 0
        sm = (eq[-1] - (eq[:-1] * solution).sum())
        solution[i] = sm / div

    return solution[:n]

def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

# %%
answer = simplex(A, b, c)

with open('simplex_answer.json', 'w+') as f:
    answer = {'answer': answer}
    json.dump(answer, f)

sys.exit(0)