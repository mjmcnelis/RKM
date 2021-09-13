#!/usr/bin/env python3
import math
import numpy as np


# standard Runge-Kutta step
def RK_standard(y, dy1, t, dt, y_prime, butcher, embedded = False):

    # y        = current solution y_n
    # dy1      = first intermediate Euler step \Delta y_n^{(1)}
    # t        = current time t_n
    # dt       = current stepsize dt_n
    # y_prime  = source function f
    # butcher  = Butcher table
    # embedded = return primary/secondary solutions (and last stage for possible FSAL) if True

    if embedded:                                            # get c_i, A_ij, b_i and number of stages from Butcher table
        c = butcher[:-2, 0]
        A = butcher[:-2, 1:]
        b = butcher[-2, 1:]
        bhat = butcher[-1, 1:]
        stages = butcher.shape[0] - 2

    else:
        c = butcher[:-1, 0]
        A = butcher[:-1, 1:]
        b = butcher[-1, 1:]
        stages = butcher.shape[0] - 1

    dy_array = [0] * stages
    dy_array[0] = dy1                                       # first intermediate Euler step

    for i in range(1, stages):                              # loop over remaining intermediate Euler steps

        dy = 0

        for j in range(0, i):

            dy += dy_array[j] * A[i,j]

        dy_array[i] = dt * y_prime(t + dt*c[i], y + dy)

    dy = 0

    for j in range(0, stages):                              # primary RK iteration (Butcher notation)

        dy += dy_array[j] * b[j]

    if embedded:                                            # secondary RK iteration (for embedded RK)

        dyhat = 0

        for j in range(0, stages):

            dyhat += dy_array[j] * bhat[j]

        k_last = dy_array[-1] / dt

        return (y + dyhat), (y + dy), k_last                # updated ERK solutions (secondary, primary, last stage)

    return y + dy                                           # updated solution (primary)



# new adaptive Runge-Kutta step
def RKM_step(y, y_prev, t, dt_prev, y_prime, method, butcher, eps, norm, dt_min, dt_max, low = 0.2, high = 1.5, flags = False):

    # y         = current solution y_n
    # y_prev    = previous solution y_{n-1}
    # t         = current time t_n
    # dt_prev   = previous step size dt_{n-1}
    # y_prime   = source function f
    # eps       = error tolerance parameter
    # norm      = type of vector norm (e.g. 1, None (2), np.inf)
    # dt_min    = min time step
    # dt_max    = max time step
    # low, high = safety bounds for dt variation (set to default values)

    order = int(method.split('_')[-1])                      # get order of method

    high = high**(1/order)                                  # suppress growth rate

    f = y_prime(t, y)                                       # for first intermediate Euler step

    y_star = y  +  dt_prev * f                              # compute y_star and approximate C w/ central differences

    C = 2 * (y_star - 2*y + y_prev) / dt_prev**2

    C_norm = np.linalg.norm(C, ord = norm)
    y_norm = np.linalg.norm(y, ord = norm)
    f_norm = np.linalg.norm(f, ord = norm)

    if C_norm == 0:                                         # prevent division by 0
        dt = dt_prev
    else:
        if (C_norm * y_norm) > (2 * eps * f_norm**2):       # compute adaptive step size
            dt = (2 * eps * y_norm / C_norm)**0.5
        else:
            dt = 2 * eps * f_norm / C_norm

        dt = min(high*dt_prev, max(low*dt_prev, dt))        # control rate of change

    dt = min(dt_max, max(dt_min, dt))                       # impose dt_min <= dt <= dt_max

    if flags and (dt == dt_min or dt == dt_max):
        print('RKM_step flag: dt = %.2e hit min/max bounds at t = %.2f (change dt_min or dt_max)' % (dt, t))

    dy1 = f * dt                                            # recycle first intermediate Euler step

    y_prev = y

    y = RK_standard(y, dy1, t, dt, y_prime, butcher)        # update y with standard Runge-Kutta method

    return y, y_prev, dt                                    # updated solution, current solution, current step size



# embedded Runge-Kutta step
def ERK_step(y0, t, dt, k1, y_prime, method, butcher, FSAL, eps, norm, dt_min, dt_max, low = 0.2, high = 5, S = 0.9, max_attempts = 100, flags = False):

    # y0           = current solution y_n
    # dt           = proposal for time step dt_n
    # k1           = first stage f(t_n, y_n)
    # S            = safety factor (set to default value)
    # max_attempts = max number of attempts

    order = int(method.split('_')[-2])                      # order of primary method
    order_hat = int(method.split('_')[-1])                  # order of secondary method

    order_max = max(order, order_hat)
    order_min = min(order, order_hat)
    power = 1 / (1 + order_min)

    high = high**(order_min / order_max)                    # adjust growth rate

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for n in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        if flags and (dt_next == dt_min or dt_next == dt_max):
            print('ERK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

        dy1 = dt * k1                                       # first intermediate Euler step

        # propose updated solution (secondary, primary, last stage)
        yhat, y, k_last = RK_standard(y0, dy1, t, dt, y_prime, butcher, embedded = True)

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

            if flags and (dt_next == dt_min or dt_next == dt_max):
                print('ERK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

            return y, k_last, dt, dt_next, n + 1            # updated solution, last stage, current dt, next dt, number of attempts

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    if flags and (dt_next == dt_min or dt_next == dt_max):
        print('ERK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

    return y, k_last, dt, dt_next, n + 1                    # return last attempt



# step doubling Runge-Kutta step
def SDRK_step(y, t, dt, y_prime, method, butcher, eps, norm, dt_min, dt_max, low = 0.2, high = 5, S = 0.9, max_attempts = 100, flags = False):

    # routine is very similar to ERK_step()

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    high = high**(order/(1+order))

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    f = y_prime(t, y)                                       # first intermediate Euler step

    for n in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        if flags and (dt_next == dt_min or dt_next == dt_max):
            print('SDRK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

        # full RK step
        dy1 = dt * f
        y1 = RK_standard(y, dy1, t, dt, y_prime, butcher, embedded = False)

        # two half RK steps
        y_mid = RK_standard(y, dy1/2, t, dt/2, y_prime, butcher, embedded = False)
        t_mid = t + dt/2
        dy1_mid = (dt/2) * y_prime(t_mid, y_mid)
        y2 = RK_standard(y_mid, dy1_mid, t_mid, dt/2, y_prime, butcher, embedded = False)

        error = (y2 - y1) / (2**order - 1)                  # estimate local truncation error
        yR = y2 + error                                     # propose updated solution (Richardson extrapolation)

        error_norm = np.linalg.norm(error, ord = norm)      # error norm
        y_norm = np.linalg.norm(yR, ord = norm)
        dy_norm = np.linalg.norm(yR - y, ord = norm)

        tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

        if error_norm == 0:
            rescale = 1                                     # prevent division by 0
        else:
            rescale = (tolerance / error_norm)**power       # scaling factor
            rescale = min(high, max(low, S*rescale))        # control rate of change

        if error_norm <= tolerance:                         # check if attempt succeeded

            dt_next = min(dt_max, max(dt_min, dt*rescale))  # impose dt_min <= dt <= dt_max

            if flags and (dt_next == dt_min or dt_next == dt_max):
                print('SDRK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

            return yR, dt, dt_next, n + 1                   # updated solution, current step size, next step size, number of attempts

    dt_next = min(dt_max, max(dt_min, dt*rescale))

    if flags and (dt_next == dt_min or dt_next == dt_max):
        print('SDRK_step flag: dt_next = %.2e hit min/max bounds at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

    return yR, dt, dt_next, n + 1                           # return last attempt



def estimate_step_size(y, t, y_prime, parameters, eps, low = 0.5, high = 10, max_attempts = 100, flags = False):

    method  = 'euler_1'                                     # use euler step doubling
    butcher = np.loadtxt('tables/butcher/standard/euler_1.dat')

    norm   = parameters['norm']                             # get parameters
    dt_min = parameters['dt_min']
    dt_max = parameters['dt_max']

    dt = dt_min
    dt_prev = dt

    order = int(method.split('_')[-1])                      # get order of method
    power = 1 / (1 + order)

    rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

    for n in range(0, max_attempts):

        dt = min(dt_max, max(dt_min, dt*rescale))           # increase step size for next attempt

        # full RK step
        dy1 = dt * y_prime(t, y)
        y1 = RK_standard(y, dy1, t, dt, y_prime, butcher, embedded = False)

        # two half RK steps
        y_mid = RK_standard(y, dy1/2, t, dt/2, y_prime, butcher, embedded = False)
        t_mid = t + dt/2
        dy1_mid = (dt/2) * y_prime(t_mid, y_mid)
        y2 = RK_standard(y_mid, dy1_mid, t_mid, dt/2, y_prime, butcher, embedded = False)

        error = (y2 - y1) / (2**order - 1)                  # estimate local truncation error

        error_norm = np.linalg.norm(error, ord = norm)      # error norm
        y_norm = np.linalg.norm(y2, ord = norm)
        dy_norm = np.linalg.norm(y2 - y, ord = norm)

        tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

        if error_norm == 0:
            rescale = high                                  # prevent division by 0
        else:
            rescale = (tolerance / error_norm)**power       # scaling factor
            rescale = min(high, max(low, rescale))

        if error_norm >= tolerance or dt >= dt_max:         # check if attempt succeeded

            return dt_prev, n + 1

        else:
            rescale = max(1.1, rescale)

        dt_prev = dt

    if flags:
        print('estimate_step_size flag: exceeded max number of attempts')

    return dt_prev, n + 1




