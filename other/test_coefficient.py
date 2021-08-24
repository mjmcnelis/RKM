#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ode_solver import ode_solver
import runge_kutta
import butcher_table
from exact_solution import y_exact, y_prime, solution, A, cycles

if solution is 'sine':                          # set initial conditions
    t0 = 0
    tf = 2*math.pi/A * cycles
    norm = 1
elif solution is 'inverse_power':
    t0 = 0.0001
    tf = 10
    norm = 2
else:
    t0 = -10
    tf = 10
    norm = 2


y0 = y_exact(t0, solution)       # todo: replace with own solution
dt0 = runge_kutta.dt_MAX


def compute_C(y_array, t_array, dt_array, norm = None):

    C_exact  = np.zeros(len(t_array)).reshape(-1,1)
    C_approx = np.zeros(len(t_array)).reshape(-1,1)

    for i in range(0, len(t_array)):
        t  = t_array[i]
        dt = dt_array[i+1]

        # compute C from exact solution
        y  = y_exact(t, solution)
        f  = y_prime(t, y, solution)

        y1 = y + dt*f
        yE = y_exact(t + dt, solution)

        C_exact[i] = (2/dt**2) * np.linalg.norm(yE - y1, ord = norm)


        # compute C from central difference approximation
        y_prev = y_array[i]
        y = y_array[i+1]
        dt_prev = dt_array[i]

        f = y_prime(t, y, solution)
        y_star = y + dt_prev*f

        C_approx[i] = (2/dt_prev**2) * np.linalg.norm(y_star - 2*y + y_prev, ord = norm)



    return C_exact, C_approx



solver = 'RKM'
method = 'DP8'

y, t, dt, evaluations, finish = ode_solver(y0, t0, tf, dt0, solver, method, eps = 1.e-8, norm = norm)

C_exact, C_approx = compute_C(y, t[1:], dt, norm = norm)

# plt.plot(t, y[:,0],  'black', label = 'Exact',      linewidth = 1.5)
plt.plot(t[1:], C_exact,  'black', label = 'Exact',      linewidth = 1.5)
plt.plot(t[1:], C_approx, 'red',   label = method + 'M', linewidth = 1.5, linestyle = 'dashed')
plt.ylabel(r'$||C||_%d$' % norm, fontsize=14)
plt.xlabel('t', fontsize=14)
plt.xlim(t0, tf)
# if solution is not 'sine':
#     plt.yscale('log')
plt.xticks(np.round(np.linspace(t0, tf, 5), 2))
plt.legend(fontsize = 14, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
plt.tick_params(labelsize = 10)
plt.tight_layout()
plt.show()




