#!/usr/bin/env python3
import math
import numpy as np
from precision import precision                     # for myfloat
from scipy import integrate

myfloat = type(precision(1))                        # todo: can I define these elsewhere? (for running ode solver)

solution = 'projectile_damped'                             # solution type

A = 100                                             # for sine function
cycles = 20

B = 0.5                                             # for logistic function

h = 0                                               # for projectile motion (y-direction)
v = 100                                             # h = initial position, v = initial velocity
g = 10                                              # g = gravitational acceleration, k = damping constant
k = 0.5

t0 = -10                                            # initial and final times
tf = 10

if solution == 'sine':
    t0 = 0
    tf = 2*math.pi/A * cycles
elif solution == 'inverse_power':
    t0 = 0.0001
elif solution in ['projectile', 'projectile_damped']:
    t0 = 0
    tf = 20



# exact solution
def y_exact(t):
    if solution == 'logistic':

        return np.array([math.exp(t) / (1 + math.exp(t)) - B], dtype = myfloat).reshape(-1,1)

    elif solution == 'gaussian':

        return np.array([math.exp(-t**2)], dtype = myfloat).reshape(-1,1)

    elif solution == 'inverse_power':

        return np.array([1/t**2], dtype = myfloat).reshape(-1,1)

    elif solution == 'sine':

        return np.array([math.sin(A*t), A*math.cos(A*t)], dtype = myfloat).reshape(-1,1)

    elif solution == 'exponential':

        return np.array([math.exp(10*t)], dtype = myfloat).reshape(-1,1)

    elif solution == 'projectile':

        return np.array([h + v*(t-t0) - 0.5*g*(t-t0)**2 , v - g*(t-t0)], dtype = myfloat).reshape(-1,1)

    elif solution == 'projectile_damped':

        return np.array([h - g*(t-t0)/k + (v + g/k)/k*(1 - math.exp(-k*(t-t0))), -g/k + (v + g/k)*math.exp(-k*(t-t0))], dtype = myfloat).reshape(-1,1)



# dy/dt = f of exact solution
def y_prime(t, y):
    if solution is 'logistic':

        return (y + B) * (1 - y - B)

    elif solution is 'gaussian':

        return - 2 * t * y

    elif solution is 'inverse_power':

        return -2 * (y**1.5)

    elif solution is 'sine':

        return np.array([y[1], - A*A*y[0]], dtype = myfloat).reshape(-1,1)

    elif solution is 'exponential':

        return 10*y

    elif solution is 'projectile':

        return np.array([y[1], -g], dtype = myfloat).reshape(-1,1)

    elif solution is 'projectile_damped':

        return np.array([y[1], -g - k*y[1]], dtype = myfloat).reshape(-1,1)



# jacobian df/dy of exact solution
def jacobian(t, y):
    if solution is 'logistic':

        return 1 - 2*(y + B)

    elif solution is 'gaussian':

        return - 2 * t

    elif solution is 'inverse_power':

        return - 3 * (y**0.5)

    elif solution is 'exponential':

        return 10

    else:
        return 0



solution_dict = {'gaussian':            r"$y^{'} = -2ty$",
                 'logistic':            r"$y^{'} = (y + \frac{1}{2})(\frac{1}{2} - y)$",
                 'inverse_power':       r"$y^{'} = -2y^{3/2}$",
                 'sine':                r"$y^{''} = -%s^{2}y$" % A,
                 'exponential':         r"$y^{'} = 10y$",
                 'projectile':          r"$y^{''} = -10$",
                 'projectile_damped':   r"$y^{''} = -10 - y'/2$"}



def compute_error_of_exact_solution(t, y, y_exact, error_type = 'absolute', average = True, norm = None):

    error_array = np.zeros(len(t))

    for i in range(0, len(t)):

        Y = y_exact(t[i])

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



