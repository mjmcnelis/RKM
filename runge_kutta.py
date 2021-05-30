#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

# import precision
from exact_solution import y_prime  # todo: pass y_prime instead
import exact_solution

solution = exact_solution.solution


# todo: any way to put a macro on flags?

dt_MIN = 1.e-7
dt_MAX = 1

LOW = 0.2
HIGH = 5

# a small HIGH (1.1) doesn't help for sine

# HIGH_RKM = 1.25      # this might be too low

HIGH_RKM = 1.4
# HIGH_RKM = 5           # for inverse power



# standard RK step
def RK_standard(y0, dy1, t, dt, butcher, embedded = False):

    # todo: pass y_prime as a function
    # todo: make use of FSAL property in BS32, DP54 (for embedded = True)

    # y0       = current solution y_n
    # dy1      = first intermediate Euler step \Delta y_n^{(1)}
    # t        = current time t_n
    # dt       = current stepsize dt_n
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
def RKM_step(y, y_prev, t, dt_prev, method, butcher, eps = 1.e-2, adaptive = True, norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH_RKM):

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
        y_star = y + dt_prev*f                              # compute y_star and approximate C

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

        if dt == dt_min or dt == dt_max:
            print('RKM_step flag: dt = %.2e at t = %.2f (change dt_min, dt_max)' % (dt, t))
    else:
        dt = dt_prev

    dy1 = f * dt                                            # recycle first intermediate Euler step
    y_prev = y
    y = RK_standard(y, dy1, t, dt, butcher)                 # update y with standard Runge-Kutta method

    return y, y_prev, dt                                    # updated solution, current solution, current step size



# embedded RK step
def ERK_step(y0, t, dt, method, butcher, eps = 1.e-8, norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH, S = 0.9, max_attempts = 100):

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

        if (dt == dt_min or dt == dt_max) and rescale < 1:
            print('ERK_step flag: dt = %.2e at t = %.2f (change dt_min, dt_max)' % (dt, t))

        dy1 = dt * y_prime(t, y0, solution)

        # propose updated solution (secondary, primary)
        yhat, y = RK_standard(y0, dy1, t, dt, butcher, embedded = True)

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

            if dt_next == dt_min or dt_next == dt_max:
                print('ERK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

            return y, dt, dt_next, i + 1                    # updated solution, current step size, next step size, number of attempts
        else:
            rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    if dt_next == dt_min or dt_next == dt_max:
                print('ERK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

    return y, dt, dt_next, i + 1                            # return last attempt



# step doubling RK step
def SDRK_step(y0, t, dt, method, butcher, eps = 1.e-8, norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH, S = 0.9, max_attempts = 100):

    # note: routine is very similar to ERK_step() above

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for i in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        if (dt == dt_min or dt == dt_max) and rescale < 1:
            print('SDRK_step flag: dt = %.2e at t = %.2f (change dt_min, dt_max)' % (dt, t))

        # full RK step
        dy1 = dt * y_prime(t, y0, solution)
        y1 = RK_standard(y0, dy1, t, dt, butcher, embedded = False)

        # two half RK steps
        y_mid = RK_standard(y0, dy1/2, t, dt/2, butcher, embedded = False)
        t_mid = t + dt/2
        dy1_mid = (dt/2) * y_prime(t_mid, y_mid, solution)
        y2 = RK_standard(y_mid, dy1_mid, t_mid, dt/2, butcher, embedded = False)

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

            if dt_next == dt_min or dt_next == dt_max:
                print('SDRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

            return y, dt, dt_next, i + 1                    # updated solution, current step size, next step size, number of attempts
        else:
            rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    if dt_next == dt_min or dt_next == dt_max:
        print('SDRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

    return y, dt, dt_next, i + 1                            # return last attempt








def estimate_step_size(y0, t, method, butcher, eps = 1.e-8, norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = 0.5, high = 10, max_attempts = 100):

    dt = dt_min
    dt_prev = dt

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for i in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # increase step size for next attempt

        # full RK step
        dy1 = dt * y_prime(t, y0, solution)
        y1 = RK_standard(y0, dy1, t, dt, butcher, embedded = False)

        # two half RK steps
        y_mid = RK_standard(y0, dy1/2, t, dt/2, butcher, embedded = False)
        t_mid = t + dt/2
        dy1_mid = (dt/2) * y_prime(t_mid, y_mid, solution)
        y2 = RK_standard(y_mid, dy1_mid, t_mid, dt/2, butcher, embedded = False)

        error = (y2 - y1) / (2**order - 1)                  # estimate local truncation error
        y = y2 + error                                      # propose updated solution (Richardson extrapolation)

        error_norm = np.linalg.norm(error, ord = norm)      # error norm
        y_norm = np.linalg.norm(y, ord = norm)
        dy_norm = np.linalg.norm(y - y0, ord = norm)

        tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

        if error_norm == 0:
            rescale = high                                  # prevent division by 0
        else:
            rescale = (tolerance / error_norm)**power       # scaling factor
            rescale = min(high, max(low, rescale))

        if error_norm >= tolerance or dt >= dt_max:         # check if attempt succeeded
            break
        else:
            rescale = max(1.1, rescale)

        dt_prev = dt

    return dt_prev, i+1



def estimate_first_step_size(y_0, t_0, solver, method, norm = None):

    # borrowed this algorithm from Hairer book (didn't work for Gaussian)

    if solver is 'ERK':                                     # get order of method
        order = int(method.split('_')[-2])
    else:
        order = int(method.split('_')[-1])

    power = 1 / (1 + order)

    f_0 = y_prime(t_0, y_0, solution)

    d_0 = np.linalg.norm(y_0, ord = norm)
    d_1 = np.linalg.norm(f_0, ord = norm)

    if d_0 < 1.e-5 or d_1 < 1.e-5:                            # first guess for dt
        dt_0 = 1.e-6
    else:
        dt_0 = 0.01 * d_0 / d_1

    y_1 = y_0  +  dt_0 * f_0                                        # computer intermediate Euler step

    f_1 = y_prime(t_0 + dt_0, y_1, solution)

    d_2 = np.linalg.norm(f_1 - f_0, ord = norm) / dt_0          # estimate for 2nd derivative

    if max(d_1, d_2) <= 1.e-15:
        dt_1 = max(1.e-6, d_0/1000)
    else:
        dt_1 = (0.01 / max(d_1, d_2))**power

    dt = min(100 * dt_0, dt_1)                                # estimate for first dt

    return dt
















