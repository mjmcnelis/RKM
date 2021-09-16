#!/usr/bin/env python3
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from explicit_runge_kutta import RK_standard, RKM_step, ERK_step, SDRK_step, estimate_step_size
from implicit_runge_kutta import DIRK_standard, EDIRK_step, SDDIRK_step
from linear_multistep import adams_bashforth, adams_moulton, adams_bashforth_moulton, differentiation_formula, compute_DF_predictor
from exact_solution import y_exact, compute_error_of_exact_solution
from dictionaries import ode_method_dict
from precision import myfloat



def get_method_name(method_label):

    return ode_method_dict[method_label][0]                     # get filename from dictionary



def get_butcher_table(adaptive, method):

    if adaptive == 'ERK':                                       # table directory
        table_dir = 'tables/butcher/embedded/'
    else:
        table_dir = 'tables/butcher/standard/'

    return np.loadtxt(table_dir + method + '.dat')              # read butcher table



def remove_extra_stage(butcher):

    stages = butcher.shape[0] - 2                               # get number of stages

    if butcher[-2,-1] == 0:                                     # remove extraneous stage
        butcher = np.delete(butcher, -3, axis = 0)
        butcher = np.delete(butcher, -1, axis = 1)              # also delete last column (should be all zeroes)
        removed_stage = True

        return butcher, True
    else:                                                       # otherwise return original table
        return butcher, False



def replace_embedded_pair(butcher, method, free):

    method_free = method.rsplit('_', 1)[0] + '_' + str(free)    # replace order of embedded pair in string

    if free == 1:                                               # free embedded pair is Euler step

        butcher[-1,1] = 1
        butcher[-1,2:] = 0

    elif free == 2:                                             # free embedded pair is generic 2nd-order method

        c1 = butcher[1,0]                                       # c1 coefficient of second stage

        if c1 == 0:
            print('replace_embedded_pair error: c1 coefficient is zero')
            quit()

        butcher[-1,1] = 1 - 1/(2*c1)
        butcher[-1,2] = 1/(2*c1)
        butcher[-1,3:] = 0

    while True:                                                 # also remove extraneous stages

        # examine extraneous stages consecutively (starting with last stage)
        # removal process may not catch all extraneous stages
        # but usually it's the case that they are the last one or two

        butcher, removed_stage = remove_extra_stage(butcher)

        if not removed_stage:
            break

    if method == 'bogacki_shampine_3_2':                        # only exception (standard table is ralston_3)
        butcher_standard = np.loadtxt('tables/butcher/standard/ralston_3.dat')
    else:                                                       # assume this file exists, if not add to butcher_table.py
        butcher_standard = np.loadtxt('tables/butcher/standard/' + method.rsplit('_', 1)[0] + '.dat')

    if not np.array_equal(butcher_standard, butcher[:-1,:]):    # for debugging stage removal
        print('replace_embedded_pair error: did not remove extraneous stages correctly')
        quit()

    return butcher, method_free



def get_multistep_table(solver, method):

    if solver == 'adams_bashforth':
        table_dir = 'tables/multistep/adams/adams_bashforth/'
    elif solver == 'adams_moulton':
        table_dir = 'tables/multistep/adams/adams_moulton/'
    elif solver == 'adams_bashforth_moulton':
        table_dir = 'tables/multistep/adams/adams_bashforth_moulton/'
    elif solver == 'backward_differentiation_formula':
        table_dir = 'tables/multistep/backward_differentiation_formula/'
    elif solver == 'numerical_differentiation_formula':
        table_dir = 'tables/multistep/numerical_differentiation_formula/'
    else:
        print('get_multistep_table error: %s is not a multistep solver' % solver)
        quit()

    table = np.loadtxt(table_dir + method + '.dat')             # read multistep table

    if method in ['adams_bashforth_1', 'adams_moulton_1']:      # for some reason, they have no shape when filed opened
        table = table.reshape(1)
    elif method == 'adams_bashforth_moulton_1':
        table = table.reshape(2,1)

    return table



def get_multistep_past_steps(table, solver):

    if solver in ['adams_bashforth', 'adams_moulton']:
        return table.shape[0] - 1
    elif solver == 'adams_bashforth_moulton':
        return table.shape[1] - 1
    else:
        return table.shape[1] - 2



def solver_is_multistep(method_label):

    if method_label[:-1] in ['AB', 'AM', 'ABM', 'BDF', 'NDF']:
        return True
    else:
        return False



def method_is_FSAL(butcher, adaptive):

    if adaptive == 'ERK':
        if np.array_equal(butcher[-3,1:], butcher[-2,1:]):      # check if second-to-last and third-to-last rows match
            return True
        else:
            return False
    else:
        if np.array_equal(butcher[-2,1:], butcher[-1,1:]):      # check if last and second-to-last rows match
            return True
        else:
            return False



def get_stages(butcher, adaptive):

    stages = butcher.shape[0] - 1                               # stages in RKM or standard RK (fixed dt)

    if adaptive == 'SDRK':
        return (3*stages - 1)                                   # stages in SDRK
    elif adaptive == 'ERK':

        if method_is_FSAL(butcher, adaptive):                   # todo: see if any standard methods have FSAL
            return (stages - 2)                                 # stages in ERK
        else:
            return (stages - 1)

    return stages



def get_order(adaptive, method):

    if adaptive == 'ERK':                                       # order of primary method
        order = int(method.split('_')[-2])                      # split string
    else:
        order = int(method.split('_')[-1])                      # order of standard method

    return order



def get_runge_kutta_solver(butcher, adaptive):

    if adaptive == 'ERK':
        A = butcher[:-2, 1:]                                    # get A_ij block of butcher table
    else:
        A = butcher[:-1, 1:]

    lower_triangular = np.allclose(A, np.tril(A))               # check whether matrix is lower triangular

    if lower_triangular:

        diag_A = np.diag(A)                                     # diagonal of A_ij

        zero_diagonal = np.array_equal(diag_A, np.zeros(len(diag_A)))

        if zero_diagonal:                                       # check whether diagonal elements are all zero
            solver = 'explicit_runge_kutta'
        else:
            solver = 'diagonal_implicit_runge_kutta'
    else:
        solver = 'fully_implicit_runge_kutta'

    return solver



def get_explicit_stages(butcher, adaptive):

    if adaptive == 'ERK':
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



def rescale_epsilon(eps, adaptive, order, method):

    # comment for implicit RKM runs (testing)
    if adaptive == 'RKM':                                       # todo: look into rescaling SDRK's eps by Richardson factor
        eps = eps**(2/(1+order))
        # eps = eps**(1/order)

    elif adaptive == 'ERK':
        order_hat = int(method.split('_')[-1])                  # order of secondary method
        order_min = min(order, order_hat)
        order_max = max(order, order_hat)

        # eps = eps**(order_min/order_max)

    return eps                                                  # use for implicit routines (testing)



# main function
def ode_solver(y0, t0, tf, y_prime, parameters, jacobian = None):

    # y0           = initial solution
    # t0           = initial time
    # tf           = final time
    # y_prime      = source function f
    # parameters   = parameter list
    # jacobian     = jacobian of source function df/dy

    start = time.time()                                         # start timer

    adaptive       = parameters['adaptive']                     # get parameters
    method_label   = parameters['method']
    dt0            = parameters['dt_initial']
    n_max          = parameters['max_steps']
    eps            = parameters['eps']
    norm           = parameters['norm']
    dt_min         = parameters['dt_min']
    dt_max         = parameters['dt_max']
    root           = parameters['root']
    max_iterations = parameters['max_iterations']
    eps_root       = parameters['eps_root']
    free_embedded  = parameters['free_embedded']

    y = y0.copy()                                               # set initial conditions
    t = t0
    dt = dt0

    y_prev = y0.copy()                                          # for RKM
    dt_next = dt                                                # for ERK/SDRK

    y_array  = np.empty(shape = [0], dtype = myfloat)           # construct arrays for (y, t, dt)
    t_array  = np.empty(shape = [0], dtype = myfloat)
    dt_array = np.empty(shape = [0], dtype = myfloat)

    method = get_method_name(method_label)                      # filename corresponding to code label

    if solver_is_multistep(method_label):                       # solver is multistep

        solver     = method[:-2]
        multistep  = get_multistep_table(solver, method)
        past_steps = get_multistep_past_steps(multistep, solver)

        method_RK = 'ssp_ketcheson_4'                           # any way to un-hardcode this?
        butcher_RK = get_butcher_table('', method_RK)
        stages_RK = get_stages(butcher_RK, '')

        f_list = []                                             # for (AB, AM, ABM) routines
        y_list = []                                             # for (BDF, NDF) routines
        dy_0 = 0

        if adaptive != None:
            print('ode_solver flag: multistep routines currently use a fixed stepsize')

    else:                                                       # solver is Runge-Kutta

        butcher = get_butcher_table(adaptive, method)           # read in butcher table from file
        solver  = get_runge_kutta_solver(butcher, adaptive)     # get Runge-Kutta solver (explicit, diagonal implicit, fully implicit)
        order   = get_order(adaptive, method)                   # get order of method

        if adaptive == 'ERK' and solver == 'explicit_runge_kutta' and order > 2 and free_embedded in [1,2]:
            butcher, method = replace_embedded_pair(butcher, method, free_embedded)
            high = 1.5
        else:
            high = 5.0                                          # default value for high parameter in ERK

        stages  = get_stages(butcher, adaptive)                 # get number of stages
        stage_explicit = get_explicit_stages(butcher, adaptive) # mark explicit stages (for DIRK_standard)
        FSAL = method_is_FSAL(butcher, adaptive)                # check if method has FSAL property

        if adaptive != None:
            eps = rescale_epsilon(eps, adaptive, order, method) # rescale epsilon_0, depending on adaptive method

    if solver not in ['explicit_runge_kutta', 'adams_bashforth'] and root != 'fixed_point' and jacobian == None:
        print('ode_solver error: need to pass jacobian for implicit routines using \'%s\' root finder' % root)
        quit()

    evaluations = 0
    total_attempts = 0
    finish = False

    for n in range(0, n_max):                                   # start evolution loop

        y_array = np.append(y_array, y).reshape(-1, y.shape[0]) # append (y,t) arrays
        t_array = np.append(t_array, t)

        if solver == 'explicit_runge_kutta':                    # explicit Runge-Kutta routines

            if adaptive == None:                                # use a fixed step size

                dy1 = dt * y_prime(t, y)
                y = RK_standard(y, dy1, t, dt, y_prime, butcher)

                evaluations += stages
                total_attempts += 1

            elif adaptive == 'RKM':                             # new adaptive time step algorithm

                if n == 0:                                      # adjust initial time step w/ euler step-doubling

                    dt_next, tries_SD = estimate_step_size(y, t, y_prime, parameters)
                    evaluations += 2 * tries_SD

                    dt = dt_next                                # then use standard Runge-Kutta step
                    dy1 = dt * y_prime(t, y)
                    y = RK_standard(y, dy1, t, dt, y_prime, butcher)

                else:                                           # use RKM for remainder

                    y, y_prev, dt = RKM_step(y, y_prev, t, dt, y_prime, method, butcher, parameters, eps)

                evaluations += stages
                total_attempts += 1

            elif adaptive == 'ERK':                             # embedded Runge-Kutta algorithm

                if FSAL and n > 0:
                    k1 = k_last                                 # recycle FSAL stage from previous step
                else:
                    k1 = y_prime(t, y)                          # compute first stage

                y, k_last, dt, dt_next, tries = ERK_step(y, t, dt_next, k1, y_prime, method, butcher, FSAL, parameters, eps, high = high)

                if FSAL:                                        # need to count FSAL stage if step rejected
                    evaluations += (stages  +  (tries - 1) * stages)
                else:
                    evaluations += (stages  +  (tries - 1) * (stages - 1))

                total_attempts += tries

            elif adaptive == 'SDRK':                            # step doubling Runge-Kutta algorithm

                y, dt, dt_next, tries = SDRK_step(y, t, dt_next, y_prime, method, butcher, parameters, eps)

                evaluations += (stages  +  (tries - 1) * (stages - 1))
                total_attempts += tries

        elif solver == 'diagonal_implicit_runge_kutta':         # diagonal implicit Runge-Kutta routines

            if adaptive == None:                                # use a fixed step size

                y, evals = DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, norm, root, max_iterations, eps_root)

                evaluations += evals
                total_attempts += 1

            if adaptive == 'RKM':                               # new adaptive step size algorithm

                if n == 0:                                      # adjust initial time step w/ step-doubling

                    # todo: clear this up

                    # method_SD = 'backward_euler_1'              # adjust initial step size w/ step-doubling
                    # butcher_SD = get_butcher_table('SDRK', method_SD)
                    # stage_explicit_SD = np.zeros(1)

                    # todo: put in step-doubling implicit euler?
                    # dt_next, evals = implicit_runge_kutta.estimate_step_size(y, t, dt_next, y_prime, jacobian, method_SD, butcher_SD,
                    #                                                          stage_explicit_SD, root = root, eps = eps/2, norm = norm)
                    # evaluations += evals

                    # temp: use explicit version
                    dt_next, tries_SD = estimate_step_size(y, t, y_prime, parameters)
                    evaluations += 2 * tries_SD

                    dt = dt_next

                    y, evals = DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, norm, root, max_iterations, eps_root)

                else:

                    y, y_prev, dt, evals = DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, norm, root, max_iterations, eps_root, 
                                                         eps = eps, adaptive = 'RKM', method = method, y_prev = y_prev)
                evaluations += evals
                total_attempts += 1

            elif adaptive == 'ERK':                             # embedded Runge-Kutta algorithm

                y, dt, dt_next, tries, evals = EDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit,
                                                          eps, norm, dt_min, dt_max, root, max_iterations, eps_root)
                evaluations += evals
                total_attempts += tries

            elif adaptive == 'SDRK':                            # step doubling Runge-Kutta algorithm

                y, dt, dt_next, tries, evals = SDDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit,
                                                           eps, norm, dt_min, dt_max, root, max_iterations, eps_root)
                evaluations += evals
                total_attempts += tries

        elif solver == 'fully_implicit_runge_kutta':            # fully implicit Runge-Kutta routines

            print('ode_solver error: no fully implicit routine yet')
            quit()

        else:                                                   # linear multistep routines

            if n < past_steps:                                  # use Runge-Kutta routine for first few steps

                f = y_prime(t, y)
                dy1 = f * dt

                f_list.insert(0, f)
                y_list.insert(0, y)

                y = RK_standard(y, dy1, t, dt, y_prime, butcher_RK)

                total_attempts += 1
                evaluations += stages_RK

            else:                                               # use multistep routine for remainder

                if solver == 'adams_bashforth':                 # adams-bashforth (explicit)

                    y, f, evals = adams_bashforth(y, t, dt, f_list, y_prime, multistep, past_steps)

                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                elif solver == 'adams_moulton':                 # adams-moulton (implicit)

                    y, f, evals = adams_moulton(y, t, dt, f_list, y_prime, jacobian, multistep, past_steps, norm, root, max_iterations, eps_root)

                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                elif solver == 'adams_bashforth_moulton':       # adams-bashforth-moulton (predictor-corrector, implicit)

                    y, f, evals = adams_bashforth_moulton(y, t, dt, f_list, y_prime, jacobian, multistep, past_steps, norm, root, max_iterations, 
                                                          eps_root)
                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                else:                                           # backward/numerical differentiation formula (predictor-corrector, implicit)

                    y, y_prev, evals = differentiation_formula(y, t, dt, y_list, y_prime, jacobian, multistep, past_steps, norm, root,
                                                               max_iterations, eps_root, dy_0 = dy_0)
                    y_list.insert(0, y_prev)

                    if solver == 'backward_differentiation_formula':
                        dy_0 = compute_DF_predictor(y, y_list, multistep)
                    else:
                        dy_0 = compute_DF_predictor(y, y_list, multistep[:,1:])    # todo: can I compute before update?

                    y_list.pop()

                    total_attempts += 1
                    evaluations += evals

        dt_array = np.append(dt_array, dt)

        if t >= tf:                                             # stop evolution
            finish = True
            break

        t += dt

    rejection_rate = 100 * (1 - (n+1)/total_attempts)           # percentage of attempts that were rejected

    if finish:
        print('\node_solver took %.2g seconds to finish\t\trejection rate = %.1f %%' % (time.time() - start, rejection_rate))
    else:
        print('\node_solver flag: evolution stopped at t = %.3g seconds' % t)

    return y_array, t_array, dt_array, evaluations, rejection_rate



# compute average error vs function evaluations (todo: move to exact solution?)
def method_efficiency(y0, t0, tf, y_prime, parameters, adaptive, method_label, eps_array, norm, error_type, average_error, jacobian = None):

    error_array = np.zeros(len(eps_array)).reshape(-1,1)
    evaluations_array = np.zeros(len(eps_array)).reshape(-1,1)

    if adaptive == None:
        adaptive_label = 'standard'
    else:
        adaptive_label = adaptive

    print('Testing efficiency of %s %s' % (adaptive_label, method_label))

    parameters['adaptive'] = adaptive                           # override parameters
    parameters['method'] = method_label
    parameters['norm'] = norm

    for i in range(0, len(eps_array)):

        eps = eps_array[i]

        parameters['eps'] = eps

        if adaptive == None:
            parameters['dt_initial'] = eps                      # replace fixed time step with eps

        y, t, dt, evaluations, reject = ode_solver(y0, t0, tf, y_prime, parameters, jacobian = jacobian)

        error = compute_error_of_exact_solution(t, y, y_exact, error_type = error_type, average_error = average_error, norm = norm)

        error_array[i] = error
        evaluations_array[i] = evaluations

    print()

    return error_array, evaluations_array









