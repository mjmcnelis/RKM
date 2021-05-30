#!/usr/bin/env python3
import math
import numpy as np
from precision import precision                     # for myfloat
from scipy import integrate

myfloat = type(precision(1))                        # todo: can I define these elsewhere? (for running ode solver)

solution = 'inverse_power'                                   # solution type

A = 100                                             # for sin(At) function
cycles = 20

B = 0.5                                             # for logistic function

# exact solution
def y_exact(t, solution):
    if solution is 'logistic':
        return np.array([math.exp(t) / (1 + math.exp(t)) - B], dtype = myfloat).reshape(-1,1)
    elif solution is 'gaussian':
        return np.array([math.exp(-t*t)], dtype = myfloat).reshape(-1,1)
    elif solution is 'inverse_power':
        return np.array([1/t**2], dtype = myfloat).reshape(-1,1)
    elif solution is 'sine':
        return np.array([math.sin(A*t), A*math.cos(A*t)], dtype = myfloat).reshape(-1,1)
    elif solution is 'exponential':
        return np.array([math.exp(10*t)], dtype = myfloat).reshape(-1,1)


# dy/dt of exact solution
def y_prime(t, y, solution):
    if solution is 'logistic':
        return (y + B) * (1 - y - B)
    elif solution is 'gaussian':
        return - 2 * t * y
    elif solution is 'inverse_power':
        return -2*(y**1.5)
    elif solution is 'sine':
        return np.array([y[1], - A*A*y[0]], dtype = myfloat).reshape(-1,1)
    elif solution is 'exponential':
        return 10*y


solution_dict = {'gaussian':        r"$y^{'} = -2ty$",
                 'logistic':        r"$y^{'} = (y + \frac{1}{2})(\frac{1}{2} - y)$",
                 'inverse_power':   r"$y^{'} = -2y^{3/2}$",
                 'sine':            r"$y^{''} = -%s^{2}y$" % A,
                 'exponential':     r"$y^{'} = 10y$"}


# todo: move to exact solution
def compute_error_of_exact_solution(t, y, solution, error_type = 'absolute', average = True, norm = None):

    error_array = np.zeros(len(t))

    for i in range(0, len(t)):

        Y = y_exact(t[i], solution)

        error = np.linalg.norm(y[i].reshape(-1,1) - Y, ord = norm)

        if error_type is 'relative':
            y_exact_norm = np.linalg.norm(Y, ord = norm)

            if y_exact_norm != 0:
                error /= y_exact_norm

        error_array[i] = error

    if average:
        error = integrate.simps(error_array, x = t) / (t[-1] - t[0])    # average error over time interval
    else:
        error = max(error_array)                                        # take the max value instead

    return error



