#!/usr/bin/env python3
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ode_solver import method_efficiency
from exact_solution import t0, tf, solution, y_exact, y_prime, jacobian, solution_dict
from parameters import parameters

if len(sys.argv) > 1:                           # run efficiency test for p = order
    order = int(sys.argv[1])
else:
    order = np.inf

y0 = y_exact(t0)

adaptive_0 = None
adaptive_1 = 'RKM'
adaptive_2 = 'ERK'
adaptive_3 = 'SDRK'

if solution is 'gaussian':                      # parameters for computing error
    norm = None
    average_error = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 5.e+2

elif solution is 'logistic':
    norm = None
    average_error = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+2

elif solution is 'inverse_power':
    norm = None
    average_error = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

elif solution is 'sine':
    norm = 1
    average_error = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+3

elif solution is 'exponential':
    norm = None
    average_error = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

else:
    print('test_efficiency error: no error type for %s set' % solution)
    quit()



# RK2 performance
#---------------------------------------------------------------------------
method_RK2_0 = 'H2'
method_RK2_1 = 'H2'
method_RK2_2 = 'HE21'
method_RK2_3 = 'H2'

if solution is 'logistic':
    method_3 = 'R2'

if order == 0 or order == 2:
    if solution is 'gaussian':
        eps_0 = 10**np.arange(-1.6, -4., -1)
        eps_1 = 10**np.arange(-2.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-2., -7., -1)
        eps_3 = 4 * 10**np.arange(-2.4, -9., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-6.5, -14., -1)
        eps_2 = 4 * 10**np.arange(-4.9, -10., -1)
        eps_3 = 4 * 10**np.arange(-6.5, -13., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-4.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 4 * 10**np.arange(-3.9, -8.9, -0.5)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4., -10., -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 10**np.arange(-3., -9., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-3.6, -9., -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 4 * 10**np.arange(-3.8, -8., -1)

    # overriding parameters: adaptive, method, eps and norm
    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_0, method_RK2_0, eps_0, norm, error_type, average_error)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_1, method_RK2_1, eps_1, norm, error_type, average_error)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_2, method_RK2_2, eps_2, norm, error_type, average_error)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_3, method_RK2_3, eps_3, norm, error_type, average_error)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2.dat',   np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat', np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK2_D.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_M.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_E.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK2_D.dat')

error_RK2_0, evals_RK2_0 = efficiency_0[:,0], efficiency_0[:,1]
error_RK2_1, evals_RK2_1 = efficiency_1[:,0], efficiency_1[:,1]
error_RK2_2, evals_RK2_2 = efficiency_2[:,0], efficiency_2[:,1]
error_RK2_3, evals_RK2_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK3 peformance
#---------------------------------------------------------------------------
method_RK3_0 = 'H3'
method_RK3_1 = 'H3'
method_RK3_2 = 'BS32'
method_RK3_3 = 'H3'

if solution is 'inverse_power':
    method_RK3_1 = 'R3'
    method_RK3_3 = 'R3'

if order == 0 or order == 3:
    if solution is 'gaussian':
        # eps_0 = 10**np.arange(-0.4, -4., -1)
        eps_0 = 10**np.arange(-1.4, -4., -1)
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.8, -8.8, -1)
        eps_3 = 10**np.arange(-2.1, -11., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8., -17., -1)
        eps_2 = 10**np.arange(-6.7, -14., -1)
        eps_3 = 10**np.arange(-7., -16., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.5, -9.5, -1)
        eps_3 = 10**np.arange(-3.6, -10.6, -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -12., -1)
        eps_2 = 10**np.arange(-3.5, -10., -1)
        eps_3 = 10**np.arange(-4.1, -11., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-5., -10., -1)
        eps_3 = 10**np.arange(-3.9, -11., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_0, method_RK3_0, eps_0, norm, error_type, average_error)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_1, method_RK3_1, eps_1, norm, error_type, average_error)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_2, method_RK3_2, eps_2, norm, error_type, average_error)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_3, method_RK3_3, eps_3, norm, error_type, average_error)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3.dat',   np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat', np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK3_D.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_M.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_E.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK3_D.dat')

error_RK3_0, evals_RK3_0 = efficiency_0[:,0], efficiency_0[:,1]
error_RK3_1, evals_RK3_1 = efficiency_1[:,0], efficiency_1[:,1]
error_RK3_2, evals_RK3_2 = efficiency_2[:,0], efficiency_2[:,1]
error_RK3_3, evals_RK3_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK4 peformance
#---------------------------------------------------------------------------
method_RK4_0 = 'F4'
method_RK4_1 = 'F4'
method_RK4_2 = 'F45'
method_RK4_3 = 'RK4'

if order == 0 or order == 4:
    if solution is 'gaussian':
        eps_0 = 10**np.arange(-1.35, -3.85, -0.5)
        eps_1 = 10**np.arange(-2.5, -12., -1)
        eps_2 = 10**np.arange(-4., -14., -1)
        eps_3 = 10**np.arange(-3.1, -13., -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8.5, -20., -1)
        eps_2 = 10**np.arange(-8.8, -18.8, -1)
        eps_3 = 10**np.arange(-8., -16., -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-3., -12., -1)
        eps_2 = 10**np.arange(-4.5, -13.5, -1)
        eps_3 = 10**np.arange(-5.1, -12.1, -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -13., -1)
        eps_2 = 10**np.arange(-5.5, -14., -1)
        # eps_2 = 10**np.arange(-1., -6., -1)
        eps_3 = 10**np.arange(-4.5, -14., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -12., -1)
        eps_2 = 10**np.arange(-5.5, -13., -1)
        eps_3 = 10**np.arange(-4.6, -13., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_0, method_RK4_0, eps_0, norm, error_type, average_error)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_1, method_RK4_1, eps_1, norm, error_type, average_error)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_2, method_RK4_2, eps_2, norm, error_type, average_error)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_3, method_RK4_3, eps_3, norm, error_type, average_error)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4.dat',   np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat', np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK4_D.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_M.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_E.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK4_D.dat')

error_RK4_0, evals_RK4_0 = efficiency_0[:,0], efficiency_0[:,1]
error_RK4_1, evals_RK4_1 = efficiency_1[:,0], efficiency_1[:,1]
error_RK4_2, evals_RK4_2 = efficiency_2[:,0], efficiency_2[:,1]
error_RK4_3, evals_RK4_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK5 peformance
#---------------------------------------------------------------------------
method_RK5_0 = 'BS5'
method_RK5_1 = 'BS5'
method_RK5_2 = 'BS54'
method_RK5_3 = 'BS5'

if solution is 'gaussian':
    method_RK5_3 = 'CK5'
elif solution is 'exponential':
    method_RK5_3 = 'CK5'
elif solution is 'logistic':
    method_RK5_1 = 'BS5'
    method_RK5_2 = 'T54'
    method_RK5_3 = 'CK5'

if order == 0 or order == 5:
    if solution is 'gaussian':
        eps_0 = 10**np.arange(-1.2, -3.2, -0.5)
        eps_1 = 10**np.arange(-2., -12., -1)
        eps_2 = 10**np.arange(-5.3, -13., -0.5)
        eps_3 = 10**np.arange(-3.9, -13.9, -1)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-8.6, -20., -1)
        eps_2 = 10**np.arange(-9., -15., -1)
        eps_3 = 10**np.arange(-8.6, -15.6, -1)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(-2., -12., -1)
        eps_2 = 10**np.arange(-6., -13., -1)
        eps_3 = 10**np.arange(-5.2, -15.2, -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.5, -13.5, -1)
        eps_2 = 10**np.arange(-7., -15., -1)
        eps_3 = 10**np.arange(-6., -16., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -12., -1)
        eps_2 = 10**np.arange(-7.2, -13.2, -0.5)
        eps_3 = 10**np.arange(-5.7, -13.7, -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_0, method_RK5_0, eps_0, norm, error_type, average_error)
    # error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_1, method_RK5_1, eps_1, norm, error_type, average_error)
    # error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_2, method_RK5_2, eps_2, norm, error_type, average_error)
    # error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_3, method_RK5_3, eps_3, norm, error_type, average_error)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5.dat',   np.concatenate((error_0, evals_0), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat', np.concatenate((error_1, evals_1), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat', np.concatenate((error_2, evals_2), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK5_D.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_M.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_E.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK5_D.dat')

error_RK5_0, evals_RK5_0 = efficiency_0[:,0], efficiency_0[:,1]
error_RK5_1, evals_RK5_1 = efficiency_1[:,0], efficiency_1[:,1]
error_RK5_2, evals_RK5_2 = efficiency_2[:,0], efficiency_2[:,1]
error_RK5_3, evals_RK5_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK8 peformance
#---------------------------------------------------------------------------
method_RK8_0 = 'DP8'
method_RK8_1 = 'DP8'
method_RK8_2 = 'DP87'
method_RK8_3 = 'SP8'

if solution is 'sine':
    method_RK8_1 = 'SP8'
elif solution is 'gaussian':
    method_RK8_3 = 'C8'
elif solution is 'logistic':
    method_RK8_1 = 'SP8'
    method_RK8_3 = 'C8'
elif solution is 'inverse_power':
    method_RK8_1 = 'SP8'
    method_RK8_3 = 'C8'

if order == 0 or order == 8:
    if solution is 'gaussian':
        eps_0 = 10**np.arange(-1.1, -2.6, -0.5)
        eps_1 = 10**np.arange(-2., -12., -1)
        eps_2 = 10**np.arange(-6.5, -15., -0.5)
        eps_3 = 10**np.arange(-5., -14., -0.5)
    elif solution is 'logistic':
        eps_1 = 10**np.arange(-1., -22., -1)
        eps_2 = 10**np.arange(-10., -17., -1)
        eps_3 = 10**np.arange(-8.6, -15., -0.5)
    elif solution is 'inverse_power':
        eps_1 = 10**np.arange(0., -15., -1)
        eps_2 = 10**np.arange(-6., -16., -1)
        eps_3 = 10**np.arange(-6.1, -16., -1)
    elif solution is 'sine':
        eps_1 = 10**np.arange(-4.8, -12.8, -1)
        eps_2 = 10**np.arange(-8.5, -15.5, -0.5)
        eps_3 = 10**np.arange(-7., -15., -1)
    elif solution is 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-8.5, -14.5, -0.5)
        eps_3 = 10**np.arange(-7., -14., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_0, method_RK8_0, eps_0, norm, error_type, average_error)
    # error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_1, method_RK8_1, eps_1, norm, error_type, average_error)
    # error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_2, method_RK8_2, eps_2, norm, error_type, average_error)
    # error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive_3, method_RK8_3, eps_3, norm, error_type, average_error)

    np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8.dat',   np.concatenate((error_0, evals_0), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat', np.concatenate((error_1, evals_1), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat', np.concatenate((error_2, evals_2), axis = 1))
    # np.savetxt('efficiency_plots/' + solution + '/data/efficiency_RK8_D.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_M.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_E.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/data/efficiency_RK8_D.dat')

error_RK8_0, evals_RK8_0 = efficiency_0[:,0], efficiency_0[:,1]
error_RK8_1, evals_RK8_1 = efficiency_1[:,0], efficiency_1[:,1]
error_RK8_2, evals_RK8_2 = efficiency_2[:,0], efficiency_2[:,1]
error_RK8_3, evals_RK8_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# efficiency plot
#---------------------------------------------------------------------------
plt.figure(figsize = (5,5))

plt.plot(evals_RK2_0, error_RK2_0, 'red',         label = method_RK2_0,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK2_1, error_RK2_1, 'red',         label = method_RK2_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK2_2, error_RK2_2, 'red',         label = method_RK2_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK2_3, error_RK2_3, 'red',         label = method_RK2_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_RK3_0, error_RK3_0, 'darkorange',  label = method_RK3_0,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK3_1, error_RK3_1, 'darkorange',  label = method_RK3_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK3_2, error_RK3_2, 'darkorange',  label = method_RK3_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK3_3, error_RK3_3, 'darkorange',  label = method_RK3_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_RK4_0, error_RK4_0, 'gold',        label = method_RK4_0,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK4_1, error_RK4_1, 'gold',        label = method_RK4_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK4_2, error_RK4_2, 'gold',        label = method_RK4_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK4_3, error_RK4_3, 'gold',        label = method_RK4_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_RK5_0, error_RK5_0, 'forestgreen', label = method_RK5_0,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK5_1, error_RK5_1, 'forestgreen', label = method_RK5_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK5_2, error_RK5_2, 'forestgreen', label = method_RK5_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK5_3, error_RK5_3, 'forestgreen', label = method_RK5_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_RK8_0, error_RK8_0, 'blue',        label = method_RK8_0,        linewidth = 1.5, alpha = 0.5)
plt.plot(evals_RK8_1, error_RK8_1, 'blue',        label = method_RK8_1 + 'M',  linewidth = 1.5)
plt.plot(evals_RK8_2, error_RK8_2, 'blue',        label = method_RK8_2,        linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_RK8_3, error_RK8_3, 'blue',        label = method_RK8_3 + 'SD', linestyle = 'dotted', linewidth = 1.5)

plt.xscale('log')
plt.yscale('log')
plt.xlim(evals_min, 1.e+5)
plt.ylim(1.e-14, 1.e+0)
plt.tick_params(labelsize = 10)
plt.title(solution_dict[solution], fontsize = 12)
plt.ylabel(error_label, fontsize = 12)
plt.xlabel('function evaluations', fontsize = 12)
plt.legend(fontsize = 9.5, borderpad = 1, labelspacing = 0, handlelength = 1.4, handletextpad = 0.5, frameon = False)
plt.tight_layout()
plt.savefig('efficiency_plots/' + solution + '/' + solution + '_efficiency.png', dpi = 200)
# plt.show()









