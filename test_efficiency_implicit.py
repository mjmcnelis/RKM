#!/usr/bin/env python3
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ode_solver import method_efficiency
from explicit_runge_kutta import dt_MAX
from exact_solution import y_exact, y_prime, jacobian, solution, solution_dict, A, cycles

if len(sys.argv) > 1:
    order = int(sys.argv[1])
else:
    order = np.inf

t0 = -10
tf = 10

y0 = y_exact(t0)
dt0 = dt_MAX

adaptive_0 = None
adaptive_1 = 'RKM'
adaptive_2 = 'ERK'
adaptive_3 = 'SDRK'

# parameters for computing error
if solution == 'logistic':
    norm = None
    average = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+2

elif solution == 'exponential':
    norm = None
    average = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3


else:
    print('test_efficiency error: no error type for %s set' % solution)
    quit()

# RK2 peformance
method_RK2_0 = 'QZ2'
method_RK2_1 = 'QZ2'
method_RK2_2 = 'CN21'
method_RK2_3 = 'BE1'

root = 'newton_fast'

if order == 0 or order == 2:

    if solution == 'logistic':

        eps_0 = 10**np.arange(0., -3., -0.5)      # fixed time step (todo: pass different dt0 values instead)


        eps_1 = 10**np.arange(-2.8, -7., -1)
        eps_2 = 10**np.arange(-3.1, -8.1, -1)
        eps_3 = 10**np.arange(-1.8, -7., -1)

    elif solution == 'exponential':
        eps_1 = 10**np.arange(-1., -4., -1)
        # eps_1 = 10**np.arange(0., -3., -0.5)      # fixed time step
        eps_2 = 10**np.arange(-1., -4., -1)
        eps_3 = 10**np.arange(-3., -5., -1)

    n_max = 100000

    error_RK2_0, evals_RK2_0 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK2_0, eps_0, error_type, adaptive = adaptive_0, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)
    error_RK2_1, evals_RK2_1 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK2_1, eps_1, error_type, adaptive = adaptive_1, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)
    error_RK2_2, evals_RK2_2 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK2_2, eps_2, error_type, adaptive = adaptive_2, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)
    error_RK2_3, evals_RK2_3 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK2_3, eps_3, error_type, adaptive = adaptive_3, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2.dat',   np.concatenate((error_RK2_0, evals_RK2_0), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_M.dat', np.concatenate((error_RK2_1, evals_RK2_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_E.dat', np.concatenate((error_RK2_2, evals_RK2_2), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_D.dat', np.concatenate((error_RK2_3, evals_RK2_3), axis=1))

efficiency_RK2_0 = np.loadtxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2.dat')
efficiency_RK2_1 = np.loadtxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_M.dat')
efficiency_RK2_2 = np.loadtxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_E.dat')
efficiency_RK2_3 = np.loadtxt('efficiency_plots/' + solution + '/implicit/data/efficiency_RK2_D.dat')

error_RK2_0, evals_RK2_0 = efficiency_RK2_0[:,0], efficiency_RK2_0[:,1]
error_RK2_1, evals_RK2_1 = efficiency_RK2_1[:,0], efficiency_RK2_1[:,1]
error_RK2_2, evals_RK2_2 = efficiency_RK2_2[:,0], efficiency_RK2_2[:,1]
error_RK2_3, evals_RK2_3 = efficiency_RK2_3[:,0], efficiency_RK2_3[:,1]





method_RK3_1 = 'LIIICS4'
method_RK3_2 = 'LIIICS42'
method_RK3_3 = 'DIRKL3'

root = 'newton'

if order == 0 or order == 3:
    if solution == 'logistic':
        eps_1 = 10**np.arange(-2.5, -7.5, -1)
        eps_2 = 10**np.arange(-3.5, -10.5, -1)
        eps_3 = 10**np.arange(-1., -12., -1)

    n_max = 100000

    error_RK3_1, evals_RK3_1 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK3_1, eps_1, error_type, adaptive = adaptive_1, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)
    error_RK3_2, evals_RK3_2 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK3_2, eps_2, error_type, adaptive = adaptive_2, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)
    error_RK3_3, evals_RK3_3 = method_efficiency(y0, t0, tf, dt0, y_prime, method_RK3_3, eps_3, error_type, adaptive = adaptive_3, jacobian = jacobian, root = root, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat', np.concatenate((error_RK3_1, evals_RK3_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat', np.concatenate((error_RK3_2, evals_RK3_2), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_D.dat', np.concatenate((error_RK3_3, evals_RK3_3), axis=1))

efficiency_RK3_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat')
efficiency_RK3_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat')
efficiency_RK3_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_D.dat')

error_RK3_1, evals_RK3_1 = efficiency_RK3_1[:,0], efficiency_RK3_1[:,1]
error_RK3_2, evals_RK3_2 = efficiency_RK3_2[:,0], efficiency_RK3_2[:,1]
error_RK3_3, evals_RK3_3 = efficiency_RK3_3[:,0], efficiency_RK3_3[:,1]





plt.figure(figsize = (5,5))
plt.plot(evals_RK2_0, error_RK2_0, 'red',         label = method_RK2_1,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK2_1, error_RK2_1, 'red',         label = method_RK2_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK2_2, error_RK2_2, 'red',         label = method_RK2_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK2_3, error_RK2_3, 'red',         label = method_RK2_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)
#
# plt.plot(evals_RK3_1, error_RK3_1, 'darkorange',  label = method_RK3_1 + 'M',  linewidth = 1.5)
# plt.plot(evals_RK3_2, error_RK3_2, 'darkorange',  label = method_RK3_2,        linestyle = 'dashed', linewidth = 1.5)
# plt.plot(evals_RK3_3, error_RK3_3, 'darkorange',  label = method_RK3_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)
#
plt.xscale('log')
plt.yscale('log')
#
plt.xlim(evals_min, 1.e+5)
plt.ylim(1.e-14, 1.e+0)
#
plt.tick_params(labelsize = 10)
#
plt.title(solution_dict[solution], fontsize = 12)
plt.ylabel(error_label, fontsize = 12)
plt.xlabel('function evaluations', fontsize = 12)
#
plt.legend(fontsize = 9.5, borderpad = 1, labelspacing = 0, handlelength = 1.4, handletextpad = 0.5, frameon = False)
plt.tight_layout()
# plt.savefig('efficiency_plots/' + solution + '/' + solution + '_efficiency.png', dpi = 200)
plt.savefig('efficiency_plots/implicit_efficiency_' + solution + '.png', dpi = 200)
# plt.show()

