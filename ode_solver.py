#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

from precision import precision     # for myfloat
import runge_kutta
from exact_solution import y_prime  # todo: pass y_prime instead
import exact_solution as exact

myfloat = type(precision(1))
solution = exact.solution



def get_butcher_table(solver, method):

    if solver is 'ERK':                                 # table directory
        table_dir = 'butcher_tables/embedded/'
    else:
        table_dir = 'butcher_tables/standard/'

    return np.loadtxt(table_dir + method + '.dat')      # read butcher table



def method_is_FSAL(butcher):
    if np.array_equal(butcher[-3,1:], butcher[-2,1:]):  # check second and third to last rows
        return True
    else:
        return False



def get_stages(butcher, solver, method):

    stages = butcher.shape[0] - 1

    if solver is 'SDRK':
        return 3*stages - 1                             # stages in SDRK
    elif solver is 'ERK':
        if method_is_FSAL(butcher):
            return stages - 2                           # stages in ERK
        else:
            return stages - 1                           # note: do not use FSAL to save time yet

    return stages                                       # stages in RKM



def get_order(solver, method):

    if solver is 'ERK':                                 # order of primary method
        order = int(method.split('_')[-2])              # split string
    else:
        order = int(method.split('_')[-1])              # order of standard method

    return order



def rescale_epsilon(eps, solver, order):

    if solver is 'RKM':                                 # todo: look into rescaling SDRK's eps by Richardson factor
        eps = eps**(2/(1+order))

    return eps



# todo: axe return steps (don't really need it)
def ode_solver(y0, t0, tf, dt0, solver, method, norm = None, eps = 1.e-8, n_max = 10000):

    # y0     = initial solution
    # t0     = initial time
    # tf     = final time
    # dt0    = initial step size
    # solver = type of ODE solver (RKM, ERK, SDRK)
    # method = Runge-Kutta method
    # eps    = error tolerance
    # high   = safety upper bound for dt growth rate
    # n_max  = max number of evolution steps

    y = y0                                                      # set initial conditions
    t = t0
    dt = dt0

    y_prev = y0                                                 # for RKM
    dt_next = dt                                                # for ERK/SDRK

    y_array  = np.empty(shape = [0], dtype = myfloat)           # construct arrays for (y, t, dt)
    t_array  = np.empty(shape = [0], dtype = myfloat)
    dt_array = np.empty(shape = [0], dtype = myfloat)

    butcher = get_butcher_table(solver, method)                 # read in butcher table
    order = get_order(solver, method)                           # get order of method

    eps = rescale_epsilon(eps, solver, order)                   # rescale epsilon_0

    attempts = 0
    evaluations = 0
    finish = False

    for n in range(0, n_max):                                   # start evolution loop

        y_array = np.append(y_array, y).reshape(-1, y.shape[0]) # append arrays
        t_array = np.append(t_array, t)

        # RKM
        if solver is 'RKM':
            tries = 1

            if n == 0:
                method_SD = 'euler_1'                            # estimate first dt w/ step-doubling
                butcher_SD = get_butcher_table(solver, method_SD)
                y_SD, dt, dt_next, tries_SD = runge_kutta.SDRK_step(y, t, dt_next, method_SD, butcher_SD, eps = eps/2)
                evaluations += 2 * tries_SD

                dt = dt_next                                    # then use standard RK
                dy1 = dt * y_prime(t, y, solution)
                y = runge_kutta.RK_standard(y, dy1, t, dt, method, butcher)
            else:
                y, y_prev, dt = runge_kutta.RKM_step(y, y_prev, t, dt, method, butcher, eps = eps)

        # embedded
        elif solver is 'ERK':
            y, dt, dt_next, tries = runge_kutta.ERK_step(y, t, dt_next, method, butcher, eps = eps)

        # step-doubling
        elif solver is 'SDRK':
            y, dt, dt_next, tries = runge_kutta.SDRK_step(y, t, dt_next, method, butcher, eps = eps)

        dt_array = np.append(dt_array, dt)

        if t >= tf:                                             # stop evolution
            finish = True
            break

        t += dt
        attempts += tries

    evaluations += attempts * get_stages(butcher, solver, method)   # get function evaluations

    return y_array, t_array, dt_array, evaluations, finish





# compute average error vs function evaluations
def method_efficiency(y0, t0, tf, dt0, solver, method, eps_array, error_type, norm = None, average = True, high = 1.5, n_max = 10000):

    error_array = np.zeros(len(eps_array)).reshape(-1,1)
    evaluations_array = np.zeros(len(eps_array)).reshape(-1,1)

    print('Testing efficiency of %s %s' % (solver, method))

    for i in range(0, len(eps_array)):

        eps = eps_array[i]

        y, t, dt, evaluations, finish = ode_solver(y0, t0, tf, dt0, solver, method, norm = norm, eps = eps, n_max = n_max)

        error = exact.compute_error_of_exact_solution(t, y, solution, error_type = error_type, average = average, norm = norm)

        if finish:
            print('eps =', '{:.1e}'.format(eps), 'finished')
        else:
            print('eps =', '{:.1e}'.format(eps), 'did not finish')

        error_array[i] = error
        evaluations_array[i] = evaluations

    print('done\n')

    return error_array, evaluations_array


