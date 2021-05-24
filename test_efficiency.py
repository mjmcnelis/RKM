#!/usr/bin/env python3
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict

from ode_solver import method_efficiency
import runge_kutta
import butcher_table
import precision
import exact_solution                           # for solution

if len(sys.argv) > 1:
    order = int(sys.argv[1])
else:
    order = np.inf

solution = exact_solution.solution

standard_dict = butcher_table.standard_dict     # combine dictionaries (todo: make a function)
embedded_dict = butcher_table.embedded_dict

total_dict = standard_dict.copy()
total_dict.update(embedded_dict)

if solution is 'sine':                          # set initial conditions
    A = exact_solution.A                        # can bring back precision()
    cycles = exact_solution.cycles
    t0 = 0
    tf = 2*math.pi/A * cycles
elif solution is 'inverse_power':
    t0 = 0.0001
    tf = 10
else:
    t0 = -10
    tf = 10

y0 = exact_solution.y_exact(t0, solution)       # todo: replace with own solution
dt0 = runge_kutta.dt_MAX

solver_1 = 'RKM'
solver_2 = 'ERK'


# parameters for computing error
if solution is 'gaussian':
    norm = None
    average = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

elif solution is 'logistic':
    norm = None
    average = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+2

elif solution is 'inverse_power':
    norm = None
    average = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

elif solution is 'sine':
    norm = 1
    average = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+3

elif solution is 'exponential':
    norm = None
    average = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

else:
    print('test_efficiency error: no error type for %s set' % solution)
    exit()


# RK2 peformance
method_RK2_1 = 'heun_2'
method_RK2_2 = 'heun_euler_2_1'

if order == 0 or order == 2:
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-2., -7., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-6.5, -14., -1)
        eps_2 = 4 * 10**np.arange(-4.9, -10., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-4.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4., -10., -1)
        eps_2 = 4 * 10**np.arange(-4., -7., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-3.6, -9., -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)

    n_max = 100000

    error_RK2_1, evals_RK2_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK2_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK2_2, evals_RK2_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK2_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat', np.concatenate((error_RK2_1, evals_RK2_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat', np.concatenate((error_RK2_2, evals_RK2_2), axis=1))

efficiency_RK2_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat')
efficiency_RK2_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat')

error_RK2_1, evals_RK2_1 = efficiency_RK2_1[:,0], efficiency_RK2_1[:,1]
error_RK2_2, evals_RK2_2 = efficiency_RK2_2[:,0], efficiency_RK2_2[:,1]



# RK3 peformance
method_RK3_1 = 'heun_3'
method_RK3_2 = 'bogacki_shampine_3_2'

if solution is 'inverse_power':
    method_RK3_1 = 'ralston_3'


if order == 0 or order == 3:
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.8, -8.8, -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8., -17., -1)
        eps_2 = 10**np.arange(-6.7, -14., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.5, -9.5, -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -12., -1)
        eps_2 = 10**np.arange(-3.5, -10., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-5., -10., -1)

    n_max = 100000

    error_RK3_1, evals_RK3_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK3_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK3_2, evals_RK3_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK3_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

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
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2.5, -12., -1)
        eps_2 = 10**np.arange(-4., -14., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8.5, -20., -1)
        eps_2 = 10**np.arange(-9., -17., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-3., -12., -1)
        eps_2 = 10**np.arange(-4.5, -13.5, -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -13., -1)
        eps_2 = 10**np.arange(-5.5, -15., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -12., -1)
        eps_2 = 10**np.arange(-5.5, -13., -1)

        # eps_1 = 10**np.arange(-4., -12., -0.1)
        # eps_2 = 10**np.arange(-5.5, -13., -0.1)

    n_max = 100000

    error_RK4_1, evals_RK4_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK4_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK4_2, evals_RK4_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK4_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat', np.concatenate((error_RK4_1, evals_RK4_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat', np.concatenate((error_RK4_2, evals_RK4_2), axis=1))

efficiency_RK4_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat')
efficiency_RK4_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat')

error_RK4_1, evals_RK4_1 = efficiency_RK4_1[:,0], efficiency_RK4_1[:,1]
error_RK4_2, evals_RK4_2 = efficiency_RK4_2[:,0], efficiency_RK4_2[:,1]



# RK5 peformance
method_RK5_1 = 'cash_karp_5'
method_RK5_2 = 'cash_karp_5_4'

if solution is 'inverse_power':
    method_RK5_1 = 'butcher_5'

if order == 0 or order == 5:
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2.8, -11.8, -0.5)
        eps_2 = 10**np.arange(-5., -13., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-9.2, -21., -0.5)
        eps_2 = 10**np.arange(-9.5, -15.5, -0.5)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-4., -12., -0.5)
        eps_2 = 10**np.arange(-5.8, -13., -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-5., -15., -1)
        eps_2 = 10**np.arange(-6.3, -15., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4.5, -13., -1)
        eps_2 = 10**np.arange(-6.5, -14., -1)

    n_max = 100000

    error_RK5_1, evals_RK5_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK5_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK5_2, evals_RK5_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK5_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat', np.concatenate((error_RK5_1, evals_RK5_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat', np.concatenate((error_RK5_2, evals_RK5_2), axis=1))

efficiency_RK5_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat')
efficiency_RK5_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat')

error_RK5_1, evals_RK5_1 = efficiency_RK5_1[:,0], efficiency_RK5_1[:,1]
error_RK5_2, evals_RK5_2 = efficiency_RK5_2[:,0], efficiency_RK5_2[:,1]



# RK6 peformance
method_RK6_1 = 'verner_6'
method_RK6_2 = 'verner_6_5'

if solution is 'inverse_power':
    method_RK6_1 = 'butcher_6'

if order == 0 or order == 6:
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2., -12., -0.5)
        eps_2 = 10**np.arange(-4., -12.5, -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8.5, -23., -1)
        eps_2 = 10**np.arange(-9.2, -15., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-2., -14., -1)
        eps_2 = 10**np.arange(-4., -14., -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-5., -13., -1)
        eps_2 = 10**np.arange(-6.4, -13., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4.3, -11., -1)
        eps_2 = 10**np.arange(-5.9, -11., -1)

    n_max = 100000

    error_RK6_1, evals_RK6_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK6_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK6_2, evals_RK6_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK6_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK6_M.dat', np.concatenate((error_RK6_1, evals_RK6_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK6_E.dat', np.concatenate((error_RK6_2, evals_RK6_2), axis=1))

efficiency_RK6_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK6_M.dat')
efficiency_RK6_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK6_E.dat')

error_RK6_1, evals_RK6_1 = efficiency_RK6_1[:,0], efficiency_RK6_1[:,1]
error_RK6_2, evals_RK6_2 = efficiency_RK6_2[:,0], efficiency_RK6_2[:,1]



# RK8 peformance
method_RK8_1 = 'dormand_prince_8'
method_RK8_2 = 'dormand_prince_8_7'

if solution is 'inverse_power':
    method_RK8_1 = 'shanks_8'

if order == 0 or order == 8:
    if solution is 'gaussian':
        eps_1 = 10**np.arange(-2., -12., -1)
        eps_2 = 10**np.arange(-6.5, -15., -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-1., -22., -1)
        eps_2 = 10**np.arange(-10., -16., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(0., -15., -1)
        eps_2 = 10**np.arange(-6., -16., -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -12., -1)
        eps_2 = 10**np.arange(-8.5, -15.5, -0.5)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-8.5, -14., -1)

        # eps_1 = 10**np.arange(-4., -11., -0.1)
        # eps_2 = 10**np.arange(-8.5, -14., -0.1)

    n_max = 10000

    error_RK8_1, evals_RK8_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK8_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK8_2, evals_RK8_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK8_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat', np.concatenate((error_RK8_1, evals_RK8_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat', np.concatenate((error_RK8_2, evals_RK8_2), axis=1))

efficiency_RK8_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat')
efficiency_RK8_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat')

error_RK8_1, evals_RK8_1 = efficiency_RK8_1[:,0], efficiency_RK8_1[:,1]
error_RK8_2, evals_RK8_2 = efficiency_RK8_2[:,0], efficiency_RK8_2[:,1]



# RK10 peformance
method_RK10_1 = 'feagin_10'
method_RK10_2 = 'feagin_10_8'

if order == 10:
    if solution is 'gaussian' or solution is 'exponential':
        eps_1 = 10**np.arange(-5., -11., -0.5)
        eps_2 = 10**np.arange(-8.2, -13.7, -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-12., -22., -1)
        eps_2 = 10**np.arange(-6., -16., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-5., -11., -0.5)
        eps_2 = 10**np.arange(-8.2, -13.7, -0.5)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-6., -12., -0.5)
        eps_2 = 10**np.arange(-9., -15., -1)

    n_max = 10000

    error_RK10_1, evals_RK10_1 = method_efficiency(y0, t0, tf, dt0, solver_1, method_RK10_1, eps_1, error_type, norm = norm, average = average, n_max = n_max)
    error_RK10_2, evals_RK10_2 = method_efficiency(y0, t0, tf, dt0, solver_2, method_RK10_2, eps_2, error_type, norm = norm, average = average, n_max = n_max)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK10_M.dat', np.concatenate((error_RK10_1, evals_RK10_1), axis=1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK10_E.dat', np.concatenate((error_RK10_2, evals_RK10_2), axis=1))

# efficiency_RK10_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK10_M.dat')
# efficiency_RK10_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK10_E.dat')

# error_RK10_1, evals_RK10_1 = efficiency_RK10_1[:,0], efficiency_RK10_1[:,1]
# error_RK10_2, evals_RK10_2 = efficiency_RK10_2[:,0], efficiency_RK10_2[:,1]



plt.figure(figsize = (5,5))
plt.plot(evals_RK2_1, error_RK2_1, 'red', label = total_dict[method_RK2_1][1], linewidth = 1.5)
plt.plot(evals_RK2_2, error_RK2_2, 'red', label = total_dict[method_RK2_2][1], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK3_1, error_RK3_1, 'darkorange', label = total_dict[method_RK3_1][1], linewidth = 1.5)
plt.plot(evals_RK3_2, error_RK3_2, 'darkorange', label = total_dict[method_RK3_2][1], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK4_1, error_RK4_1, 'gold', label = total_dict[method_RK4_1][1], linewidth = 1.5)
plt.plot(evals_RK4_2, error_RK4_2, 'gold', label = total_dict[method_RK4_2][1], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK5_1, error_RK5_1, 'forestgreen', label = total_dict[method_RK5_1][1], linewidth = 1.5)
plt.plot(evals_RK5_2, error_RK5_2, 'forestgreen', label = total_dict[method_RK5_2][1], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK6_1, error_RK6_1, 'deepskyblue', label = total_dict[method_RK6_1][1], linewidth = 1.5)
plt.plot(evals_RK6_2, error_RK6_2, 'deepskyblue', label = total_dict[method_RK6_2][1], linestyle = 'dashed', linewidth = 1.5)
#
plt.plot(evals_RK8_1, error_RK8_1, 'blue', label = total_dict[method_RK8_1][1], linewidth = 1.5)
plt.plot(evals_RK8_2, error_RK8_2, 'blue', label = total_dict[method_RK8_2][1], linestyle = 'dashed', linewidth = 1.5)
#
# plt.plot(evals_RK10_1, error_RK10_1, 'blueviolet', label = butcher_table.methods_dict[method_RK10_1], linewidth = 1.5)
# plt.plot(evals_RK10_2, error_RK10_2, 'blueviolet', label = butcher_table.methods_dict[method_RK10_2], linestyle = 'dashed', linewidth = 1.5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(evals_min, 1.e+5)
plt.ylim(1.e-14, 1.e+0)
plt.tick_params(labelsize = 10)
plt.title(exact_solution.solution_dict[solution], fontsize = 12)
plt.ylabel(error_label, fontsize = 12)
plt.xlabel('function evaluations', fontsize = 12)
plt.legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
plt.tight_layout()
plt.savefig('efficiency_plots/' + solution + '/' + solution + '_efficiency.png', dpi = 200)



# from scipy import interpolate

# points = 101
# evals_array = np.linspace(6000, 16000, points)

# tck = interpolate.splrep(evals_RK4_1, error_RK4_1, k = 3)
# error_RK4_1_interp = interpolate.splev(evals_array, tck)

# tck= interpolate.splrep(evals_RK4_2, error_RK4_2, k = 3)
# error_RK4_2_interp = interpolate.splev(evals_array, tck)


# plt.figure(figsize=(5,5))
# plt.plot(evals_array, error_RK4_1_interp/error_RK4_2_interp, 'r', linewidth = 2.5)
# plt.show()


# tck = interpolate.splrep(evals_RK8_1, error_RK8_1, k = 3)
# error_RK8_1_interp = interpolate.splev(evals_array, tck)

# tck= interpolate.splrep(evals_RK8_2, error_RK8_2, k = 3)
# error_RK8_2_interp = interpolate.splev(evals_array, tck)


# plt.figure(figsize=(5,5))
# plt.plot(evals_array, error_RK8_1_interp/error_RK8_2_interp, 'b', linewidth = 2.5)
# plt.show()







