#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import upper
from numpy.lib.twodim_base import triu_indices_from

from precision import precision     # for myfloat
from butcher_table import standard_dict, embedded_dict
import explicit_runge_kutta
import implicit_runge_kutta
import exact_solution as exact

myfloat = type(precision(1))

# combine dictionaries
total_dict = standard_dict.copy()
total_dict.update(embedded_dict)


def get_method_fname(method_label):
    return total_dict[method_label][0]



def get_butcher_table(adaptive, method):

    if adaptive is 'ERK':                                       # table directory
        table_dir = 'butcher_tables/embedded/'
    else:
        table_dir = 'butcher_tables/standard/'

    return np.loadtxt(table_dir + method + '.dat')              # read butcher table



def method_is_FSAL(butcher):
    if np.array_equal(butcher[-3,1:], butcher[-2,1:]):          # check second and third to last rows
        return True
    else:
        return False



def get_stages(butcher, adaptive):

    stages = butcher.shape[0] - 1                               # stages in RKM or standard RK (fixed dt)

    if adaptive is 'SDRK':
        return (3*stages - 1)                                   # stages in SDRK
    elif adaptive is 'ERK':
        if method_is_FSAL(butcher):
            return (stages - 2)                                 # stages in ERK
        else:
            return (stages - 1)                                 # note: do not use FSAL to save time yet

    return stages                                       



def get_order(adaptive, method):

    if adaptive is 'ERK':                                       # order of primary method
        order = int(method.split('_')[-2])                      # split string
    else:
        order = int(method.split('_')[-1])                      # order of standard method

    return order



def get_solver(butcher, adaptive):

    if adaptive is 'ERK':
        A = butcher[:-2, 1:]                                    # get A_ij block of butcher table
    else:
        A = butcher[:-1, 1:]                                           

    lower_triangular = np.allclose(A, np.tril(A))               # check whether matrix is lower triangular

    if lower_triangular:

        diag_A = np.diag(A)                                     # diagonal of A_ij

        zero_diagonal = np.array_equal(diag_A, np.zeros(len(diag_A)))   

        if zero_diagonal:                                       # check whether diagonal elements are all zero
            solver = 'explicit'
        else:
            solver = 'diagonal_implicit'
    else:
        solver = 'fully_implicit'

    return solver



def get_explicit_stages(butcher, adaptive):

    if adaptive is 'ERK':
        A = butcher[:-2, 1:]                                    # get A_ij block of butcher table
    else:
        A = butcher[:-1, 1:]

    stages = A.shape[0]                                         # number of stages in A_ij

    explicit_stages = np.zeros(stages)                          # 0 = implicit, 1 = explicit

    for i in range(0, stages):
        upper_right_row = A[i, i:]                              # get row, starting with diagonal entry

        upper_right_row_zero = np.array_equal(upper_right_row, np.zeros(len(upper_right_row)))

        if upper_right_row_zero:
            explicit_stages[i] = 1                              # mark stage as explicit 

    return explicit_stages



def rescale_epsilon(eps, adaptive, order):

    # comment for implicit RKM runs for now
    
    # if adaptive is 'RKM':                                       # todo: look into rescaling SDRK's eps by Richardson factor
    #     eps = eps**(2/(1+order))
    #     # eps = eps**(1/order)

    return eps



# ODE solver
def ode_solver(y0, t0, tf, dt0, y_prime, method_label, adaptive = None, jacobian = None, root = 'newton_fast', norm = None, eps = 1.e-8, n_max = 10000):

    # y0           = initial solution
    # t0           = initial time
    # tf           = final time
    # dt0          = proposed initial step size 
    # y_prime      = source function f
    # adaptive     = adaptive scheme
    # method_label = Runge-Kutta method's code label
    # jacobian     = jacobian of source function df/dy 
    # root         = root solver for implicit RK routines
    # norm         = value of l in the l-norm
    # eps          = error tolerance parameter
    # n_max        = max number of evolution steps

    y = y0                                                      # set initial conditions
    t = t0
    dt = dt0

    y_prev = y0                                                 # for RKM
    dt_next = dt                                                # for ERK/SDRK

    y_array  = np.empty(shape = [0], dtype = myfloat)           # construct arrays for (y, t, dt)
    t_array  = np.empty(shape = [0], dtype = myfloat)
    dt_array = np.empty(shape = [0], dtype = myfloat)

    method  = get_method_fname(method_label)                    # filename corresponding to code label
    butcher = get_butcher_table(adaptive, method)               # read in butcher table from file
    order   = get_order(adaptive, method)                       # get order of method
    stages  = get_stages(butcher, adaptive)                     # get number of stages
    solver  = get_solver(butcher, adaptive)                     # get RK solver (i.e. explicit/implicit)
    stage_explicit = get_explicit_stages(butcher, adaptive)     # mark explicit stages (for DIRK_standard)      

    if solver is not 'explicit' and jacobian is None:           # add: and root is not 'fixed_point'?
        print('ode_solver error: need to pass jacobian for implicit runge kutta routines that use Newton\'s method')
        quit()

    if adaptive is None:
        dt *= eps                                               # rescale fixed time step by epsilon_0
    else:
        eps = rescale_epsilon(eps, adaptive, order)             # if use RKM, rescale epsilon_0 by order of method
 
    evaluations = 0
    total_attempts = 0
    finish = False

    # should I count evaluations of first adaptive RK step?

    for n in range(0, n_max):                                   # start evolution loop

        y_array = np.append(y_array, y).reshape(-1, y.shape[0]) # append arrays
        t_array = np.append(t_array, t)

        if solver is 'explicit':                                # explicit RK routines

            if adaptive is None:
                print('no fixed time step yet for explicit integrators')
                quit()

            # RKM
            elif adaptive is 'RKM':
                tries = 1

                if n == 0:
                    method_SD = 'euler_1'                       # adjust initial step size w/ step-doubling
                    butcher_SD = get_butcher_table('SDRK', method_SD)
                    dt_next, tries_SD = explicit_runge_kutta.estimate_step_size(y, t, y_prime, method_SD, butcher_SD, eps = eps/2, norm = norm)

                    evaluations += 2 * tries_SD

                    # print('RKM: dt = %.2g after %d attempts at n = 0' % (dt_next, tries_SD))

                    dt = dt_next                                # then use standard RK
                    dy1 = dt * y_prime(t, y)
                    y = explicit_runge_kutta.RK_standard(y, dy1, t, dt, y_prime, butcher)

                    evaluations += stages
                else:
                    y, y_prev, dt = explicit_runge_kutta.RKM_step(y, y_prev, t, dt, y_prime, method, butcher, eps = eps, norm = norm)

                    evaluations += stages
                    total_attempts += tries

            # embedded
            elif adaptive is 'ERK':
                y, dt, dt_next, tries = explicit_runge_kutta.ERK_step(y, t, dt_next, y_prime, method, butcher, eps = eps, norm = norm)

                if method_is_FSAL(butcher) and tries > 1:
                    evaluations += (stages  +  (tries - 1) * (stages + 1))
                else:
                    evaluations += (tries * stages)

                if n > 0:
                    total_attempts += tries

                # if tries > 1:
                #     print('ERK: dt = %.2g after %d attempts at t = %.2g ' % (dt, tries, t))

            # step-doubling
            elif adaptive is 'SDRK':
                y, dt, dt_next, tries = explicit_runge_kutta.SDRK_step(y, t, dt_next, y_prime, method, butcher, eps = eps, norm = norm)

                evaluations += (tries * stages)

                if n > 0:
                    total_attempts += tries

                # if tries > 1:
                #     print('SDRK: dt = %.2g after %d attempts at t = %.2g ' % (dt, tries, t))

        elif solver is 'diagonal_implicit':                     # diagonal implicit RK routines

            if adaptive is None:
                y, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, 
                                                              root = root, norm = norm) 
                if n > 0:
                    evaluations += evals

                if n > 0:
                    total_attempts += 1

            # RKM
            if adaptive is 'RKM':

                if n == 0:
                    # method_SD = 'backward_euler_1'              # adjust initial step size w/ step-doubling
                    # butcher_SD = get_butcher_table('SDRK', method_SD)
                    # stage_explicit_SD = np.zeros(1)

                    # todo: put in step-doubling implicit euler?
                    # dt_next, evals = implicit_runge_kutta.estimate_step_size(y, t, dt_next, y_prime, jacobian, method_SD, butcher_SD,     
                    #                                                          stage_explicit_SD, root = root, eps = eps/2, norm = norm) 
                    # evaluations += evals

                    # temp: use explicit version
                    method_SD = 'euler_1'                       # adjust initial step size w/ step-doubling
                    butcher_SD = get_butcher_table('SDRK', method_SD)
                    dt_next, tries_SD = explicit_runge_kutta.estimate_step_size(y, t, y_prime, method_SD, butcher_SD, eps = eps/2, norm = norm)
                    # evaluations += 2 * tries_SD

                    dt = dt_next

                    y, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit,
                                                                  root = root, eps = eps, norm = norm)
                    # evaluations += evals

                else:
                    y, y_prev, dt, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit,
                                                                              root = root, eps = eps, norm = norm,
                                                                              adaptive = 'RKM', method = method, y_prev = y_prev)
                    evaluations += evals
                    total_attempts += 1

            # embedded
            elif adaptive is 'ERK':
                y, dt, dt_next, tries, evals = implicit_runge_kutta.EDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit, 
                                                                               root = root, eps = eps, norm = norm) 
                
                if n > 0:
                    evaluations += evals

                if n > 0:
                    total_attempts += tries

            # step doubling
            elif adaptive is 'SDRK':
                y, dt, dt_next, tries, evals = implicit_runge_kutta.SDDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit, 
                                                                                root = root, eps = eps, norm = norm) 
                evaluations += evals

                if n > 0:
                    total_attempts += tries

        elif solver is 'fully_implicit':                        # fully implicit RK routines
            print('ode_solver error: no fully implicit routine yet')
            quit()

        dt_array = np.append(dt_array, dt)

        if t >= tf:                                             # stop evolution
            finish = True
            break

        t += dt

    steps = n

    rejection_rate = 100 * (1 - steps/total_attempts)           # percentage of attempts that were rejected

    return y_array, t_array, dt_array, evaluations, rejection_rate, finish



# compute average error vs function evaluations
def method_efficiency(y0, t0, tf, dt0, y_prime, method_label, eps_array, error_type, adaptive = None, jacobian = None, root = 'newton_fast', norm = None, average = True, high = 1.5, n_max = 10000):

    error_array = np.zeros(len(eps_array)).reshape(-1,1)
    evaluations_array = np.zeros(len(eps_array)).reshape(-1,1)

    print('Testing efficiency of %s %s' % (adaptive, method_label))

    for i in range(0, len(eps_array)):

        eps = eps_array[i]

        y, t, dt, evaluations, reject, finish = ode_solver(y0, t0, tf, dt0, y_prime, method_label, adaptive = adaptive, jacobian = jacobian, root = root, norm = norm, eps = eps, n_max = n_max)

        y_exact = exact.y_exact
        error = exact.compute_error_of_exact_solution(t, y, y_exact, error_type = error_type, average = average, norm = norm)

        if finish:
            print('eps =', '{:.1e}'.format(eps), 'finished\t\t rejection rate = %.1f %%' % reject)
        else:
            print('eps =', '{:.1e}'.format(eps), 'did not finish\t\t rejection rate = %.1f %%' % reject)

        error_array[i] = error
        evaluations_array[i] = evaluations

    print('done\n')

    return error_array, evaluations_array


