#!/usr/bin/env python3
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict

from ode_solver import method_efficiency
import butcher_table
import precision
import exact_solution                           # for solution

if len(sys.argv) > 1:
    order = int(sys.argv[1])
else:
    order = np.inf

solution = exact_solution.solution

if solution is 'sine':                          # set initial conditions
    A = exact_solution.A                        # can bring back precision()
    cycles = exact_solution.cycles
    t0 = 0
    tf = 2*math.pi/A * cycles
else:
    t0 = -10
    tf = 10

y0 = exact_solution.y_exact(t0, solution)       # todo: replace with own solution
dt0 = 0.01

solver_1 = 'RKM'
solver_2 = 'ERK'

if solution is 'logistic':
    error_type = 'absolute'
    error_label = 'average absolute error'
else:
    error_type = 'relative'
    error_label = 'average relative error'

if solution is 'sine':
    norm = 1
else:
    norm = None

# todo: use command arguments to run each test (2, 3, 4, 5, 8, 10)
# todo: save the data to file and open/plot below

# RK2 peformance
method_RK2_1 = 'heun_2'
method_RK2_2 = 'heun_euler_2_1'

if order == 0 or order == 2:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-2.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-2., -7., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-7., -14., -1)
        eps_2 = 4 * 10**np.arange(-5., -10., -1)

    n_max = 100000

    error_RK2_1, evals_RK2_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK2_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK2_2, evals_RK2_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK2_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat', np.concatenate((error_RK2_1, evals_RK2_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat', np.concatenate((error_RK2_2, evals_RK2_2), axis=1))

efficiency_RK2_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat')
efficiency_RK2_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat')

error_RK2_1, evals_RK2_1 = efficiency_RK2_1[:,0], efficiency_RK2_1[:,1]
error_RK2_2, evals_RK2_2 = efficiency_RK2_2[:,0], efficiency_RK2_2[:,1]



# RK3 peformance
method_RK3_1 = 'heun_3'
method_RK3_2 = 'bogacki_shampine_3_2'

if order == 0 or order == 3:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.8, -8.8, -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8.5, -17., -1)
        eps_2 = 10**np.arange(-7., -14., -1)

    n_max = 100000

    error_RK3_1, evals_RK3_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK3_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK3_2, evals_RK3_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK3_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat', np.concatenate((error_RK3_1, evals_RK3_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat', np.concatenate((error_RK3_2, evals_RK3_2), axis=1))

efficiency_RK3_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat')
efficiency_RK3_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat')

error_RK3_1, evals_RK3_1 = efficiency_RK3_1[:,0], efficiency_RK3_1[:,1]
error_RK3_2, evals_RK3_2 = efficiency_RK3_2[:,0], efficiency_RK3_2[:,1]



# RK4 peformance
method_RK4_1 = 'fehlberg_4'
method_RK4_2 = 'fehlberg_4_5'

if order == 0 or order == 4:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-2.5, -12., -1)
        eps_2 = 10**np.arange(-4., -14., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-9.5, -20., -1)
        eps_2 = 10**np.arange(-10., -17., -1)

    n_max = 100000

    error_RK4_1, evals_RK4_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK4_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK4_2, evals_RK4_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK4_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat', np.concatenate((error_RK4_1, evals_RK4_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat', np.concatenate((error_RK4_2, evals_RK4_2), axis=1))

efficiency_RK4_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat')
efficiency_RK4_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat')

error_RK4_1, evals_RK4_1 = efficiency_RK4_1[:,0], efficiency_RK4_1[:,1]
error_RK4_2, evals_RK4_2 = efficiency_RK4_2[:,0], efficiency_RK4_2[:,1]



# RK5 peformance
method_RK5_1 = 'cash_karp_5'
method_RK5_2 = 'cash_karp_5_4'

if order == 0 or order == 5:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-2.8, -11.8, -0.5)
        eps_2 = 10**np.arange(-5., -13., -1)
    else:
        eps_1 = 10**np.arange(-11., -24., -1)
        eps_2 = 10**np.arange(-10., -16., -1)

    n_max = 100000

    error_RK5_1, evals_RK5_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK5_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK5_2, evals_RK5_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK5_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat', np.concatenate((error_RK5_1, evals_RK5_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat', np.concatenate((error_RK5_2, evals_RK5_2), axis=1))

efficiency_RK5_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat')
efficiency_RK5_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat')

error_RK5_1, evals_RK5_1 = efficiency_RK5_1[:,0], efficiency_RK5_1[:,1]
error_RK5_2, evals_RK5_2 = efficiency_RK5_2[:,0], efficiency_RK5_2[:,1]



# RK6 peformance
method_RK6_1 = 'verner_6'
method_RK6_2 = 'verner_6_5'

if order == 0 or order == 6:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-2., -12., -0.5)
        eps_2 = 10**np.arange(-4., -12.5, -0.5)
    else:
        eps_1 = 10**np.arange(-11., -24., -1)
        eps_2 = 10**np.arange(-10., -16., -1)

    n_max = 100000

    error_RK6_1, evals_RK6_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK6_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK6_2, evals_RK6_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK6_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK6_M.dat', np.concatenate((error_RK6_1, evals_RK6_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK6_E.dat', np.concatenate((error_RK6_2, evals_RK6_2), axis=1))

efficiency_RK6_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK6_M.dat')
efficiency_RK6_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK6_E.dat')

error_RK6_1, evals_RK6_1 = efficiency_RK6_1[:,0], efficiency_RK6_1[:,1]
error_RK6_2, evals_RK6_2 = efficiency_RK6_2[:,0], efficiency_RK6_2[:,1]



# RK8 peformance
method_RK8_1 = 'dormand_prince_8'
method_RK8_2 = 'dormand_prince_8_7'

if order == 0 or order == 8:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-1., -12., -1)
        eps_2 = 10**np.arange(-6.3, -15.3, -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-12., -22., -1)
        eps_2 = 10**np.arange(-6., -16., -1)

    n_max = 10000

    error_RK8_1, evals_RK8_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK8_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK8_2, evals_RK8_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK8_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat', np.concatenate((error_RK8_1, evals_RK8_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat', np.concatenate((error_RK8_2, evals_RK8_2), axis=1))

efficiency_RK8_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat')
efficiency_RK8_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat')

error_RK8_1, evals_RK8_1 = efficiency_RK8_1[:,0], efficiency_RK8_1[:,1]
error_RK8_2, evals_RK8_2 = efficiency_RK8_2[:,0], efficiency_RK8_2[:,1]



# RK10 peformance
method_RK10_1 = 'feagin_10'
method_RK10_2 = 'feagin_10_8'

if order == 0 or order == 10:
    if solution is 'gaussian' or solution is 'sine':
        eps_1 = 10**np.arange(-5., -11., -0.5)
        eps_2 = 10**np.arange(-8.2, -13.7, -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-12., -22., -1)
        eps_2 = 10**np.arange(-6., -16., -1)

    n_max = 10000

    error_RK10_1, evals_RK10_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK10_1, eps_1, error_type, norm = norm, n_max = n_max)
    error_RK10_2, evals_RK10_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK10_2, eps_2, error_type, norm = norm, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK10_M.dat', np.concatenate((error_RK10_1, evals_RK10_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK10_E.dat', np.concatenate((error_RK10_2, evals_RK10_2), axis=1))

efficiency_RK10_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK10_M.dat')
efficiency_RK10_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK10_E.dat')

error_RK10_1, evals_RK10_1 = efficiency_RK10_1[:,0], efficiency_RK10_1[:,1]
error_RK10_2, evals_RK10_2 = efficiency_RK10_2[:,0], efficiency_RK10_2[:,1]



plt.figure(figsize = (5,5))
plt.plot(evals_RK2_1, error_RK2_1, 'red', label = butcher_table.methods_dict[method_RK2_1], linewidth = 1.5)
plt.plot(evals_RK2_2, error_RK2_2, 'red', label = butcher_table.methods_dict[method_RK2_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK3_1, error_RK3_1, 'darkorange', label = butcher_table.methods_dict[method_RK3_1], linewidth = 1.5)
plt.plot(evals_RK3_2, error_RK3_2, 'darkorange', label = butcher_table.methods_dict[method_RK3_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK4_1, error_RK4_1, 'gold', label = butcher_table.methods_dict[method_RK4_1], linewidth = 1.5)
plt.plot(evals_RK4_2, error_RK4_2, 'gold', label = butcher_table.methods_dict[method_RK4_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK5_1, error_RK5_1, 'forestgreen', label = butcher_table.methods_dict[method_RK5_1], linewidth = 1.5)
plt.plot(evals_RK5_2, error_RK5_2, 'forestgreen', label = butcher_table.methods_dict[method_RK5_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK6_1, error_RK6_1, 'deepskyblue', label = butcher_table.methods_dict[method_RK6_1], linewidth = 1.5)
plt.plot(evals_RK6_2, error_RK6_2, 'deepskyblue', label = butcher_table.methods_dict[method_RK6_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK8_1, error_RK8_1, 'blue', label = butcher_table.methods_dict[method_RK8_1], linewidth = 1.5)
plt.plot(evals_RK8_2, error_RK8_2, 'blue', label = butcher_table.methods_dict[method_RK8_2], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK10_1, error_RK10_1, 'blueviolet', label = butcher_table.methods_dict[method_RK10_1], linewidth = 1.5)
plt.plot(evals_RK10_2, error_RK10_2, 'blueviolet', label = butcher_table.methods_dict[method_RK10_2], linestyle = 'dashed', linewidth = 1.5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1.e+3, 1.e+5)
plt.ylim(1.e-14, 1.e+0)
plt.tick_params(labelsize = 10)
plt.title(exact_solution.solution_dict[solution], fontsize = 12)
plt.ylabel(error_label, fontsize = 12)
plt.xlabel('function evaluations', fontsize = 12)
plt.legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
plt.tight_layout()
plt.savefig('efficiency_plots/' + solution + '/' + solution + '_efficiency.png', dpi = 200)







