#!/usr/bin/env python3
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import explicit_runge_kutta
import implicit_runge_kutta
import linear_multistep
import exact_solution as exact
from dictionaries import ode_method_dict
from precision import precision                                 # for myfloat
from explicit_runge_kutta import dt_MIN

COLLISION_DETECTION = True                                      # for playing with collision detection

myfloat = type(precision(1))



def get_method_name(method_label):
    return ode_method_dict[method_label][0]                     # get filename from dictionary



def get_butcher_table(adaptive, method):

    if adaptive is 'ERK':                                       # table directory
        table_dir = 'tables/butcher/embedded/'

    else:
        table_dir = 'tables/butcher/standard/'

    return np.loadtxt(table_dir + method + '.dat')              # read butcher table



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

    elif method is 'adams_bashforth_moulton_1':
        table = table.reshape(2,1)

    return table



def get_multistep_past_steps(table, solver):

    if solver in ['adams_bashforth', 'adams_moulton']:
        return table.shape[0] - 1

    if solver == 'adams_bashforth_moulton':
        return table.shape[1] - 1

    else:
        return table.shape[1] - 2



def solver_is_multistep(method_label):

    if method_label[:-1] in ['AB', 'AM', 'ABM', 'BDF', 'NDF']:
        return True
    else:
        return False



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

    # comment for implicit RKM runs (testing)
    if adaptive == 'RKM':                                       # todo: look into rescaling SDRK's eps by Richardson factor
        eps = eps**(2/(1+order))
        # eps = eps**(1/order)

    return eps                                                # use for implicit routines (testing)



def simple_collision_detection(y, y_prev, t, dt, y_prime, butcher):

    last_position = y_prev[0,0]
    position = y[0,0]

    collision = False

    if position <= 0:                                           # check if projectile hit flat ground (height = 0)

        collision = True

        dt *= last_position / (last_position - position)        # estimate collision time t + dt

        y[0] = 0                                                # adjust coordinates via elastic collision
        y[1] *= -1

        print("hit ground at t = %.3f (dt' = %.3f)" % ((t + dt), dt))

        if dt < dt_MIN:                                         # briefly continue after collision

            y[0] += (dt - dt_MIN) * y[1]                        # assume velocity constant in remainder dt interval

            # dt_remainder = dt - dt_MIN                        # more sophisticated version
            # dy1 = dt_remainder * y_prime(t,y)
            # y = explicit_runge_kutta.RK_standard(y, dy1, t + dt, dt_remainder, y_prime, butcher)

            dt = dt_MIN

    return y, dt, collision



# main function
def ode_solver(y0, t0, tf, dt0, y_prime, method_label, adaptive = None, jacobian = None, root = 'newton_fast', norm = None, eps = 1.e-8, n_max = 10000):

    # y0           = initial solution
    # t0           = initial time
    # tf           = final time
    # dt0          = proposed initial step size
    # y_prime      = source function f
    # adaptive     = adaptive stepsize scheme (currently not applicable to multistep solvers)
    # method_label = code label of runge-kutta or multistep method
    # jacobian     = jacobian of source function df/dy
    # root         = root solver for implicit routines
    # norm         = value of l in the l-norm
    # eps          = error tolerance parameter
    # n_max        = max number of evolution steps

    start = time.time()                                         # start timer

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

    else:                                                       # solver is runge kutta

        butcher = get_butcher_table(adaptive, method)           # read in butcher table from file
        order   = get_order(adaptive, method)                   # get order of method
        stages  = get_stages(butcher, adaptive)                 # get number of stages
        solver  = get_runge_kutta_solver(butcher, adaptive)     # get runge-kutta solver (explicit, diagonal implicit, fully implicit)
        stage_explicit = get_explicit_stages(butcher, adaptive) # mark explicit stages (for DIRK_standard)

        if adaptive != None:
            eps = rescale_epsilon(eps, adaptive, order)         # if use adaptive RKM, rescale epsilon_0 by order of method

    if solver not in ['explicit_runge_kutta', 'adams_bashforth'] and root != 'fixed_point' and jacobian == None:
        print('ode_solver error: need to pass jacobian for implicit routines using \'%s\' root finder' % root)
        quit()

    evaluations = 0
    total_attempts = 0
    finish = False

    for n in range(0, n_max):                                   # start evolution loop

        y_array = np.append(y_array, y).reshape(-1, y.shape[0]) # append (y,t) arrays
        t_array = np.append(t_array, t)

        if solver == 'explicit_runge_kutta':                    # explicit runge-kutta routines

            if adaptive == None:                                # use a fixed step size

                y_prev = y.copy()

                dy1 = dt * y_prime(t, y)
                y = explicit_runge_kutta.RK_standard(y, dy1, t, dt, y_prime, butcher)

                evaluations += stages
                total_attempts += 1

            elif adaptive == 'RKM':                             # new adaptive step size algorithm

                if n == 0:

                    method_SD = 'euler_1'                       # adjust initial step size w/ step-doubling
                    butcher_SD = get_butcher_table('SDRK', method_SD)
                    dt_next, tries_SD = explicit_runge_kutta.estimate_step_size(y, t, y_prime, method_SD, butcher_SD, eps = eps/2, norm = norm)

                    evaluations += 2 * tries_SD

                    dt = dt_next                                # then use standard RK
                    dy1 = dt * y_prime(t, y)
                    y = explicit_runge_kutta.RK_standard(y, dy1, t, dt, y_prime, butcher)

                else:                                           # use RKM for remainder

                    y, y_prev, dt = explicit_runge_kutta.RKM_step(y, y_prev, t, dt, y_prime, method, butcher, eps = eps, norm = norm)

                evaluations += stages
                total_attempts += 1

            elif adaptive == 'ERK':                             # embedded runge-kutta algorithm

                y, dt, dt_next, tries = explicit_runge_kutta.ERK_step(y, t, dt_next, y_prime, method, butcher, eps = eps, norm = norm)

                if method_is_FSAL(butcher) and tries > 1:
                    evaluations += (stages  +  (tries - 1) * (stages + 1))
                else:
                    evaluations += (tries * stages)

                total_attempts += tries

            elif adaptive == 'SDRK':                            # step doubling runge-kutta algorithm

                y, dt, dt_next, tries = explicit_runge_kutta.SDRK_step(y, t, dt_next, y_prime, method, butcher, eps = eps, norm = norm)

                evaluations += (tries * stages)
                total_attempts += tries

        elif solver == 'diagonal_implicit_runge_kutta':         # diagonal implicit runge-kutta routines

            if adaptive == None:                                # use a fixed step size

                y, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, root = root, norm = norm)

                evaluations += evals
                total_attempts += 1

            if adaptive == 'RKM':                               # new adaptive step size algorithm

                if n == 0:

                    # todo: clear this up

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

                    y, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, root = root, eps = eps, norm = norm)

                else:

                    y, y_prev, dt, evals = implicit_runge_kutta.DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit,
                                                                              root = root, eps = eps, norm = norm,
                                                                              adaptive = 'RKM', method = method, y_prev = y_prev)

                evaluations += evals
                total_attempts += 1

            elif adaptive == 'ERK':                             # embedded runge-kutta algorithm

                y, dt, dt_next, tries, evals = implicit_runge_kutta.EDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit,
                                                                               root = root, eps = eps, norm = norm)

                evaluations += evals
                total_attempts += tries

            elif adaptive == 'SDRK':                            # step doubling runge-kutta algorithm

                y, dt, dt_next, tries, evals = implicit_runge_kutta.SDDIRK_step(y, t, dt_next, y_prime, jacobian, method, butcher, stage_explicit,
                                                                                root = root, eps = eps, norm = norm)

                evaluations += evals
                total_attempts += tries

        elif solver == 'fully_implicit_runge_kutta':            # fully implicit runge-kutta routines

            print('ode_solver error: no fully implicit routine yet')
            quit()

        else:                                                   # linear multistep routines

            if n < past_steps:                                  # use runge-kutta routine for first few steps

                f = y_prime(t, y)
                dy1 = f * dt

                f_list.insert(0, f)
                y_list.insert(0, y)

                y = explicit_runge_kutta.RK_standard(y, dy1, t, dt, y_prime, butcher_RK)

                total_attempts += 1
                evaluations += stages_RK

            else:                                               # use multistep routine for remainder

                if solver == 'adams_bashforth':                 # adams-bashforth (explicit)

                    y, f, evals = linear_multistep.adams_bashforth(y, t, dt, f_list, y_prime, multistep, past_steps)

                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                elif solver == 'adams_moulton':                 # adams-moulton (implicit)

                    y, f, evals = linear_multistep.adams_moulton(y, t, dt, f_list, y_prime, jacobian, multistep, past_steps, root)

                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                elif solver == 'adams_bashforth_moulton':       # adams-bashforth-moulton (predictor-corrector, implicit)

                    y, f, evals = linear_multistep.adams_bashforth_moulton(y, t, dt, f_list, y_prime, jacobian, multistep, past_steps, root)

                    f_list.insert(0, f)
                    f_list.pop()

                    total_attempts += 1
                    evaluations += evals

                else:                                           # backward/numerical differentiation formula (predictor-corrector, implicit)

                    y, y_prev, evals = linear_multistep.differentiation_formula(y, t, dt, y_list, y_prime, jacobian, multistep, past_steps, root, dy_0 = dy_0)

                    y_list.insert(0, y_prev)

                    if solver == 'backward_differentiation_formula':
                        dy_0 = linear_multistep.compute_DF_predictor(y, y_list, multistep)
                    else:
                        dy_0 = linear_multistep.compute_DF_predictor(y, y_list, multistep[:,1:])    # todo: can I compute before update?

                    y_list.pop()

                    total_attempts += 1
                    evaluations += evals



        if COLLISION_DETECTION and exact.solution in ['projectile', 'projectile_damped']:
            y, dt, collision = simple_collision_detection(y, y_prev, t, dt, y_prime, butcher)



        dt_array = np.append(dt_array, dt)

        if t >= tf:                                             # stop evolution
            finish = True
            break

        t += dt


    if finish:
        print('\node_solver took %.3g seconds to finish\n' % (time.time() - start))
    else:
        print('\node_solver flag: evolution stopped at t = %.3g seconds' % t)


    rejection_rate = 100 * (1 - (n+1)/total_attempts)           # percentage of attempts that were rejected

    return y_array, t_array, dt_array, evaluations, rejection_rate



# compute average error vs function evaluations (todo: move to exact solution?)
def method_efficiency(y0, t0, tf, dt0, y_prime, method_label, eps_array, error_type,
                      adaptive = None, jacobian = None, root = 'newton_fast', norm = None, average = True, high = 1.5, n_max = 10000):

    error_array = np.zeros(len(eps_array)).reshape(-1,1)
    evaluations_array = np.zeros(len(eps_array)).reshape(-1,1)

    print('Testing efficiency of %s %s' % (adaptive, method_label))

    for i in range(0, len(eps_array)):

        eps = eps_array[i]

        y, t, dt, evaluations, reject = ode_solver(y0, t0, tf, dt0, y_prime, method_label,
                                                   adaptive = adaptive, jacobian = jacobian, root = root, norm = norm, eps = eps, n_max = n_max)

        y_exact = exact.y_exact
        error = exact.compute_error_of_exact_solution(t, y, y_exact, error_type = error_type, average = average, norm = norm)

        error_array[i] = error
        evaluations_array[i] = evaluations

    print('done\n')

    return error_array, evaluations_array


