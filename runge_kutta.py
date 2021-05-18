#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# import precision
from exact_solution import y_prime  # todo: pass y_prime instead
import exact_solution

solution = exact_solution.solution


# todo: move this elsewhere
from collections import OrderedDict
colors = OrderedDict([
    ('blue', '#4e79a7'),
    ('orange', '#f28e2b'),
    ('green', '#59a14f'),
    ('red', '#e15759'),
    ('cyan', '#76b7b2'),
    ('purple', '#b07aa1'),
    ('brown', '#9c755f'),
    ('yellow', '#edc948'),
    ('pink', '#ff9da7'),
    ('gray', '#bab0ac')
])



# standard RK step
def RK_standard(y0, dy1, t, dt, method, butcher, embedded = False):

    # todo: pass y_prime as a function
    # todo: make use of FSAL property in BS32, DP54 (for embedded = True)

    # y0       = current solution y_n
    # dy1      = first intermediate Euler step \Delta y_n^{(1)}
    # t        = current time t_n
    # dt       = current stepsize dt_n
    # method   = Runge-Kutta method
    # butcher  = Butcher table
    # embedded = return primary/secondary solutions if True

    if embedded:                                            # get number of stages from Butcher table
        stages = butcher.shape[0] - 2
    else:
        stages = butcher.shape[0] - 1

    dy_array = [0] * stages
    dy_array[0] = dy1                                       # first intermediate Euler step

    for i in range(1, stages):                              # loop over remaining intermediate Euler steps
        dy = 0
        for j in range(0, i):
            dy += dy_array[j] * butcher[i,j+1]

        dy_array[i] = dt * y_prime(t + dt*butcher[i,0], y0 + dy, solution)

    dy = 0
    for i in range(0, stages):                              # primary RK iteration (Butcher notation)
        dy += dy_array[i] * butcher[stages,i+1]

    if embedded:                                            # secondary RK iteration (for embedded RK)
        dyhat = 0
        for i in range(0, stages):
            dyhat += dy_array[i] * butcher[stages+1,i+1]

        return (y0 + dyhat), (y0 + dy)                      # updated ERK solutions (secondary, primary)

    return y0 + dy                                          # updated solution (primary)



# my adaptive RK step
def RKM_step(y, y_prev, t, dt_prev, method, butcher, eps = 1.e-2, adaptive = True, norm = None, dt_min = 1.e-6, dt_max = 1, low = 0.5, high = 1.25):

    # y         = current solution y_n
    # y_prev    = previous solution y_{n-1}
    # t         = current time t_n
    # dt_prev   = previous step size dt_{n-1}
    # eps       = tolerance parameter
    # adaptive  = make dt adaptive if True
    # norm      = order of vector norm (e.g. 1, 2 (None), np.inf)
    # dt_min    = min step size
    # dt_max    = max step size
    # low, high = safety bounds for dt growth rate

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    high = high**power

    f = y_prime(t, y, solution)                             # for first intermediate Euler step

    if adaptive:
        y_star = y + f*dt_prev                              # compute y_star and approximate C

        C_norm = 2 * np.linalg.norm(y_star - 2*y + y_prev, ord = norm) / dt_prev**2
        y_norm = np.linalg.norm(y, ord = norm)
        f_norm = np.linalg.norm(f, ord = norm)

        if C_norm == 0:                                     # prevent division by 0
            dt = dt_prev
        else:
            if (C_norm * y_norm) > (2 * eps * f_norm**2):   # compute adaptive step size (piecewise formula)
                dt = (2 * eps * y_norm / C_norm)**0.5
            else:
                dt = 2 * eps * f_norm / C_norm

            dt = min(high*dt_prev, max(low*dt_prev, dt))    # control rate of change

        dt = min(dt_max, max(dt_min, dt))                   # impose dt_min <= dt <= dt_max
    else:
        dt = dt_prev

    dy1 = f * dt                                            # recycle first intermediate Euler step
    y_prev = y
    y = RK_standard(y, dy1, t, dt, method, butcher)         # update y with standard Runge-Kutta method

    return y, y_prev, dt                                    # updated solution, current solution, current step size



# embedded RK step
def ERK_step(y0, t, dt, method, butcher, eps = 1.e-8, norm = None, dt_min = 1.e-6, dt_max = 1, low = 0.5, high = 1.5, S = 0.9, max_attempts = 100):

    # y0           = current solution y_n
    # t            = current time t_n
    # dt           = starting step size
    # max_attempts = max number of attempts
    # S            = safety factor
    # note: remaining variables have same meaning as RKM

    order = int(method.split('_')[-2])                      # order of primary method
    order_hat = int(method.split('_')[-1])                  # order of secondary method

    order_min = min(order, order_hat)
    power = 1 / (1 + order_min)

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for i in range(0, max_attempts):
        dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt
        dy1 = dt * y_prime(t, y0, solution)

        # propose updated solution (secondary, primary)
        yhat, y = RK_standard(y0, dy1, t, dt, method, butcher, embedded = True)

        error_norm = np.linalg.norm(y - yhat, ord = norm)   # estimate local truncation error
        y_norm = np.linalg.norm(y, ord = norm)
        dy_norm = np.linalg.norm(y - y0, ord = norm)

        tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

        if error_norm == 0:
            rescale = 1                                     # prevent division by 0
        else:
            rescale = (tolerance / error_norm)**power       # scaling factor
            rescale = min(high, max(low, S*rescale))        # control rate of change

        if error_norm <= tolerance:                         # check if attempt succeeded
            dt_next = min(dt_max, max(dt_min, dt*rescale))  # impose dt_min <= dt <= dt_max
            return y, dt, dt_next, i + 1                    # updated solution, current step size, next step size, number of attempts
        else:
            rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    return y, dt, dt_next, i + 1                            # return last attempt



# step doubling RK step
def SDRK_step(y0, t, dt, method, butcher, eps = 1.e-8, norm = None, dt_min = 1.e-6, dt_max = 1, low = 0.5, high = 1.5, S = 0.9, max_attempts = 100):

    # note: routine is very similar to ERK_step() above

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for i in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        # full RK step
        dy1 = dt * y_prime(t, y0, solution)
        y1 = RK_standard(y0, dy1, t, dt, method, butcher, embedded = False)

        # two half RK steps
        y_mid = RK_standard(y0, dy1/2, t, dt/2, method, butcher, embedded = False)
        t_mid = t + dt/2
        dy1_mid = (dt/2) * y_prime(t_mid, y_mid, solution)
        y2 = RK_standard(y_mid, dy1_mid, t_mid, dt/2, method, butcher, embedded = False)

        error = (y2 - y1) / (2**order - 1)                  # estimate local truncation error
        y = y2 + error                                      # propose updated solution (Richardson extrapolation)

        error_norm = np.linalg.norm(error, ord = norm)      # error norm
        y_norm = np.linalg.norm(y, ord = norm)
        dy_norm = np.linalg.norm(y - y0, ord = norm)

        tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

        if error_norm == 0:
            rescale = 1                                     # prevent division by 0
        else:
            rescale = (tolerance / error_norm)**power       # scaling factor
            rescale = min(high, max(low, S*rescale))        # control rate of change

        if error_norm <= tolerance:                         # check if attempt succeeded
            dt_next = min(dt_max, max(dt_min, dt*rescale))  # impose dt_min <= dt <= dt_max
            return y, dt, dt_next, i + 1                    # updated solution, current step size, next step size, number of attempts
        else:
            rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    return y, dt, dt_next, i + 1                            # return last attempt




