#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# import exact_solution
from precision import precision     # for myfloat
import runge_kutta
from exact_solution import y_prime  # todo: pass y_prime instead
import exact_solution

myfloat = type(precision(1))
solution = exact_solution.solution



def get_butcher_table(solver, method):

    if solver is 'ERK':                                 # table directory
        table_dir = 'butcher_tables/embedded/'
    else:
        table_dir = 'butcher_tables/standard/'

    return np.loadtxt(table_dir + method + '.dat')      # read butcher table



def method_is_FSAL(method):
    # note: need to hard-code FSAL methods here (any others?)
    if method is 'bogacki_shampine_3_2' or method is 'dormand_prince_5_4':
        return True
    else:
        return False


def get_stages(butcher, solver, method):

    stages = butcher.shape[0] - 1

    if solver is 'SDRK':
        return 3*stages - 1                             # stages in SDRK
    elif solver is 'ERK':
        if method_is_FSAL(method):
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
def ode_solver(y0, t0, tf, dt0, solver, method, eps = 1.e-8, high = 1.5, n_max = 10000):

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
                y_SD, dt, dt_next, tries_SD = runge_kutta.SDRK_step(y, t, dt_next, method_SD, butcher_SD, eps = eps/2, high = high)
                evaluations += 2 * tries_SD

                dt = dt_next                                    # then use standard RK
                dy1 = dt * y_prime(t, y, solution)
                y = runge_kutta.RK_standard(y, dy1, t, dt, method, butcher)
            else:
                y, y_prev, dt = runge_kutta.RKM_step(y, y_prev, t, dt, method, butcher, eps = eps, high = high)

        # embedded
        elif solver is 'ERK':
            y, dt, dt_next, tries = runge_kutta.ERK_step(y, t, dt_next, method, butcher, eps = eps, high = high)

        # step-doubling
        elif solver is 'SDRK':
            y, dt, dt_next, tries = runge_kutta.SDRK_step(y, t, dt_next, method, butcher, eps = eps, high = high)

        dt_array = np.append(dt_array, dt)

        if t >= tf:                                             # stop evolution
            finish = True
            break

        t += dt
        attempts += tries

    evaluations += attempts * get_stages(butcher, solver, method)   # get function evaluations

    return y_array, t_array, dt_array, evaluations, finish





