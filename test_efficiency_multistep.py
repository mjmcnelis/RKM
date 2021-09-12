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

adaptive = None


if solution == 'gaussian':                      # parameters for computing error
    norm = None
    average_error = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 5.e+2

elif solution == 'logistic':
    norm = None
    average_error = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+2

elif solution == 'inverse_power':
    norm = None
    average_error = True
    error_type = 'relative'
    error_label = 'average relative error'
    evals_min = 1.e+3

elif solution == 'sine':
    norm = 1
    average_error = True
    error_type = 'absolute'
    error_label = 'average absolute error'
    evals_min = 1.e+3

elif solution == 'exponential':
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
method_MS2_0 = 'BDF2'
method_MS2_1 = 'NDF2'
method_MS2_2 = 'AB2'
method_MS2_3 = 'ABM2'

if order == 0 or order == 2:
    if solution == 'gaussian':
        eps_0 = 10**np.arange(-2., -4., -0.5)
        eps_1 = 10**np.arange(-2., -4., -0.5)
        eps_2 = 10**np.arange(-2.3, -4.3, -0.5)
        eps_3 = 10**np.arange(-1.8, -4.3, -0.5)
    elif solution == 'logistic':
        eps_1 = 10**np.arange(-6.5, -14., -1)
        eps_2 = 4 * 10**np.arange(-4.9, -10., -1)
        eps_3 = 4 * 10**np.arange(-6.5, -13., -1)
    elif solution == 'inverse_power':
        eps_1 = 10**np.arange(-4.25, -9.25, -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 4 * 10**np.arange(-3.9, -8.9, -0.5)
    elif solution == 'sine':
        eps_0 = 10**np.arange(-3., -5.5, -0.5)
        eps_1 = 10**np.arange(-4., -10., -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 10**np.arange(-3., -9., -1)
    elif solution == 'exponential':
        eps_1 = 10**np.arange(-3.6, -9., -1)
        eps_2 = 4 * 10**np.arange(-3., -7., -1)
        eps_3 = 4 * 10**np.arange(-3.8, -8., -1)

    # overriding parameters: adaptive, method, eps and norm
    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS2_0, eps_0, norm, error_type, average_error, jacobian = jacobian)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS2_1, eps_1, norm, error_type, average_error, jacobian = jacobian)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS2_2, eps_2, norm, error_type, average_error, jacobian = jacobian)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS2_3, eps_3, norm, error_type, average_error, jacobian = jacobian)

    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF2.dat', np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF2.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB2.dat',  np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM2.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF2.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF2.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB2.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM2.dat')

error_MS2_0, evals_MS2_0 = efficiency_0[:,0], efficiency_0[:,1]
error_MS2_1, evals_MS2_1 = efficiency_1[:,0], efficiency_1[:,1]
error_MS2_2, evals_MS2_2 = efficiency_2[:,0], efficiency_2[:,1]
error_MS2_3, evals_MS2_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK3 peformance
#---------------------------------------------------------------------------
method_MS3_0 = 'BDF3'
method_MS3_1 = 'NDF3'
method_MS3_2 = 'AB3'
method_MS3_3 = 'ABM3'

if order == 0 or order == 3:
    if solution == 'gaussian':
        eps_0 = 10**np.arange(-2., -4., -0.5)
        eps_1 = 10**np.arange(-2., -4., -0.5)
        eps_2 = 10**np.arange(-2.3, -4.3, -0.5)
        eps_3 = 10**np.arange(-1.8, -3.8, -0.5)

    elif solution == 'logistic':
        eps_1 = 10**np.arange(-8., -17., -1)
        eps_2 = 10**np.arange(-6.7, -14., -1)
        eps_3 = 10**np.arange(-7., -16., -1)
    elif solution == 'inverse_power':
        eps_1 = 10**np.arange(-2.2, -11.2, -1)
        eps_2 = 10**np.arange(-2.5, -9.5, -1)
        eps_3 = 10**np.arange(-3.6, -10.6, -1)
    elif solution == 'sine':

        eps_0 = 10**np.arange(-3., -6., -1)

        eps_1 = 10**np.arange(-4.5, -12., -1)
        eps_2 = 10**np.arange(-3.5, -10., -1)
        eps_3 = 10**np.arange(-4.1, -11., -1)
    elif solution == 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-5., -10., -1)
        eps_3 = 10**np.arange(-3.9, -11., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS3_0, eps_0, norm, error_type, average_error, jacobian = jacobian)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS3_1, eps_1, norm, error_type, average_error, jacobian = jacobian)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS3_2, eps_2, norm, error_type, average_error, jacobian = jacobian)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS3_3, eps_3, norm, error_type, average_error, jacobian = jacobian)

    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF3.dat', np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF3.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB3.dat',  np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM3.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF3.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF3.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB3.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM3.dat')

error_MS3_0, evals_MS3_0 = efficiency_0[:,0], efficiency_0[:,1]
error_MS3_1, evals_MS3_1 = efficiency_1[:,0], efficiency_1[:,1]
error_MS3_2, evals_MS3_2 = efficiency_2[:,0], efficiency_2[:,1]
error_MS3_3, evals_MS3_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK4 peformance
#---------------------------------------------------------------------------
method_MS4_0 = 'BDF4'
method_MS4_1 = 'NDF4'
method_MS4_2 = 'AB4'
method_MS4_3 = 'ABM4'

if order == 0 or order == 4:
    if solution == 'gaussian':
        eps_0 = 10**np.arange(-2., -4., -0.5)
        eps_1 = 10**np.arange(-2., -4., -0.5)
        eps_2 = 10**np.arange(-2.3, -4.3, -0.5)
        eps_3 = 10**np.arange(-1.8, -3.8, -0.5)

    elif solution == 'logistic':
        eps_1 = 10**np.arange(-8.5, -20., -1)
        eps_2 = 10**np.arange(-8.8, -18.8, -1)
        eps_3 = 10**np.arange(-8., -16., -1)
    elif solution == 'inverse_power':
        eps_1 = 10**np.arange(-3., -12., -1)
        eps_2 = 10**np.arange(-4.5, -13.5, -1)
        eps_3 = 10**np.arange(-5.1, -12.1, -1)
    elif solution == 'sine':
        eps_0 = 10**np.arange(-3., -5., -0.5)
        eps_1 = 10**np.arange(-4.5, -13., -1)
        eps_2 = 10**np.arange(-5.5, -14., -1)
        eps_3 = 10**np.arange(-4.5, -14., -1)
    elif solution == 'exponential':
        eps_1 = 10**np.arange(-4., -12., -1)
        eps_2 = 10**np.arange(-5.5, -13., -1)
        eps_3 = 10**np.arange(-4.6, -13., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS4_0, eps_0, norm, error_type, average_error, jacobian = jacobian)
    error_1, evals_1 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS4_1, eps_1, norm, error_type, average_error, jacobian = jacobian)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS4_2, eps_2, norm, error_type, average_error, jacobian = jacobian)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS4_3, eps_3, norm, error_type, average_error, jacobian = jacobian)

    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF4.dat', np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF4.dat', np.concatenate((error_1, evals_1), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB4.dat',  np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM4.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF4.dat')
efficiency_1 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_NDF4.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB4.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM4.dat')

error_MS4_0, evals_MS4_0 = efficiency_0[:,0], efficiency_0[:,1]
error_MS4_1, evals_MS4_1 = efficiency_1[:,0], efficiency_1[:,1]
error_MS4_2, evals_MS4_2 = efficiency_2[:,0], efficiency_2[:,1]
error_MS4_3, evals_MS4_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK5 peformance
#---------------------------------------------------------------------------
method_MS5_0 = 'BDF5'
method_MS5_2 = 'AB5'
method_MS5_3 = 'ABM5'

if order == 0 or order == 5:
    if solution == 'gaussian':
        eps_0 = 10**np.arange(-2., -4., -0.5)
        eps_2 = 10**np.arange(-2.3, -4.3, -0.5)
        eps_3 = 10**np.arange(-1.8, -3.8, -0.5)

    elif solution == 'logistic':
        eps_1 = 10**np.arange(-8.6, -20., -1)
        eps_2 = 10**np.arange(-9., -15., -1)
        eps_3 = 10**np.arange(-8.6, -15.6, -1)
    elif solution == 'inverse_power':
        eps_1 = 10**np.arange(-2., -12., -1)
        eps_2 = 10**np.arange(-6., -13., -1)
        eps_3 = 10**np.arange(-5.2, -15.2, -1)
    elif solution == 'sine':

        eps_0 = 10**np.arange(-3., -4.5, -0.5)

        eps_1 = 10**np.arange(-4.5, -13.5, -1)
        eps_2 = 10**np.arange(-7., -15., -1)
        eps_3 = 10**np.arange(-6., -16., -1)
    elif solution == 'exponential':
        eps_1 = 10**np.arange(-4., -12., -1)
        eps_2 = 10**np.arange(-7.2, -13.2, -0.5)
        eps_3 = 10**np.arange(-5.7, -13.7, -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS5_0, eps_0, norm, error_type, average_error, jacobian = jacobian)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS5_2, eps_2, norm, error_type, average_error, jacobian = jacobian)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS5_3, eps_3, norm, error_type, average_error, jacobian = jacobian)

    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF5.dat', np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB5.dat',  np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM5.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF5.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB5.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM5.dat')

error_MS5_0, evals_MS5_0 = efficiency_0[:,0], efficiency_0[:,1]
error_MS5_2, evals_MS5_2 = efficiency_2[:,0], efficiency_2[:,1]
error_MS5_3, evals_MS5_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# RK8 peformance
#---------------------------------------------------------------------------
method_MS6_0 = 'BDF6'
method_MS6_2 = 'AB6'
method_MS6_3 = 'ABM6'

if order == 0 or order == 6:
    if solution == 'gaussian':
        eps_0 = 10**np.arange(-2., -4., -0.5)
        eps_2 = 10**np.arange(-2.3, -4.3, -0.5)
        eps_3 = 10**np.arange(-1.8, -3.8, -0.5)

    elif solution == 'logistic':
        eps_1 = 10**np.arange(-1., -22., -1)
        eps_2 = 10**np.arange(-10., -17., -1)
        eps_3 = 10**np.arange(-8.6, -15., -0.5)
    elif solution == 'inverse_power':
        eps_1 = 10**np.arange(0., -15., -1)
        eps_2 = 10**np.arange(-6., -16., -1)
        eps_3 = 10**np.arange(-6.1, -16., -1)
    elif solution == 'sine':
        eps_0 = 10**np.arange(-2.4, -4.4, -0.5)
        eps_1 = 10**np.arange(-4.8, -12.8, -1)
        eps_2 = 10**np.arange(-8.5, -15.5, -0.5)
        eps_3 = 10**np.arange(-7., -15., -1)
    elif solution == 'exponential':
        eps_1 = 10**np.arange(-4., -11., -1)
        eps_2 = 10**np.arange(-8.5, -14.5, -0.5)
        eps_3 = 10**np.arange(-7., -14., -1)

    error_0, evals_0 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS6_0, eps_0, norm, error_type, average_error, jacobian = jacobian)
    error_2, evals_2 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS6_2, eps_2, norm, error_type, average_error, jacobian = jacobian)
    error_3, evals_3 = method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_MS6_3, eps_3, norm, error_type, average_error, jacobian = jacobian)

    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF6.dat', np.concatenate((error_0, evals_0), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB6.dat',  np.concatenate((error_2, evals_2), axis = 1))
    np.savetxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM6.dat', np.concatenate((error_3, evals_3), axis = 1))

efficiency_0 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_BDF6.dat')
efficiency_2 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_AB6.dat')
efficiency_3 = np.loadtxt('efficiency_plots/' + solution + '/multistep/data/efficiency_ABM6.dat')

error_MS6_0, evals_MS6_0 = efficiency_0[:,0], efficiency_0[:,1]
error_MS6_2, evals_MS6_2 = efficiency_2[:,0], efficiency_2[:,1]
error_MS6_3, evals_MS6_3 = efficiency_3[:,0], efficiency_3[:,1]
#---------------------------------------------------------------------------



# efficiency plot
#---------------------------------------------------------------------------
plt.figure(figsize = (5,5))

plt.plot(evals_MS2_0, error_MS2_0, 'red',         label = method_MS2_0, linewidth = 1.5, alpha = 0.5)
plt.plot(evals_MS2_1, error_MS2_1, 'red',         label = method_MS2_1, linewidth = 1.5)
plt.plot(evals_MS2_2, error_MS2_2, 'red',         label = method_MS2_2, linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_MS2_3, error_MS2_3, 'red',         label = method_MS2_3, linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_MS3_0, error_MS3_0, 'darkorange',  label = method_MS3_0, linewidth = 1.5, alpha = 0.5)
plt.plot(evals_MS3_1, error_MS3_1, 'darkorange',  label = method_MS3_1, linewidth = 1.5)
plt.plot(evals_MS3_2, error_MS3_2, 'darkorange',  label = method_MS3_2, linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_MS3_3, error_MS3_3, 'darkorange',  label = method_MS3_3, linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_MS4_0, error_MS4_0, 'gold',        label = method_MS4_0, linewidth = 1.5, alpha = 0.5)
plt.plot(evals_MS4_1, error_MS4_1, 'gold',        label = method_MS4_1, linewidth = 1.5)
plt.plot(evals_MS4_2, error_MS4_2, 'gold',        label = method_MS4_2, linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_MS4_3, error_MS4_3, 'gold',        label = method_MS4_3, linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_MS5_0, error_MS5_0, 'forestgreen', label = method_MS5_0, linewidth = 1.5, alpha = 0.5)
plt.plot(evals_MS5_2, error_MS5_2, 'forestgreen', label = method_MS5_2, linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_MS5_3, error_MS5_3, 'forestgreen', label = method_MS5_3, linestyle = 'dotted', linewidth = 1.5)

plt.plot(evals_MS6_0, error_MS6_0, 'deepskyblue', label = method_MS6_0, linewidth = 1.5, alpha = 0.5)
plt.plot(evals_MS6_2, error_MS6_2, 'deepskyblue', label = method_MS6_2, linestyle = 'dashed', linewidth = 1.5)
plt.plot(evals_MS6_3, error_MS6_3, 'deepskyblue', label = method_MS6_3, linestyle = 'dotted', linewidth = 1.5)

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
plt.savefig('efficiency_plots/' + solution + '/multistep/' + solution + '_efficiency.png', dpi = 200)
# plt.show()









