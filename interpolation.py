#!/usr/bin/env python3
import math
import numpy as np
from scipy import interpolate


def interpolate_solution(y, t, dt, method = 'cubic_spline', order = 3, points = 10):

    y  = y.astype('float64')
    t  = t.astype('float64')
    dt = dt.astype('float64')

    t0 = t[0]
    tf = t[-1]
    t_uniform  = np.linspace(t0, tf, points, dtype = np.float64)

    N = y.shape[1]
    y_int = np.zeros(N).reshape(-1,1)

    print(N)
    print(t.shape, y_int.shape)

    if method == 'cubic_spline':
    	for i in range(0, N):
    		y_int[i] = interpolate.splev(t_uniform, interpolate.splrep(t, y[:,i], k = order)).reshape(-1,1)

    	dt = interpolate.splev(t_uniform, interpolate.splrep(t, dt, k = order)).reshape(-1,1)
    else:
        print('nothing else yet')

    print(y_int[0])
    quit()

    return y_int, t_uniform, dt






# old notes
# from scipy import interpolate

# points = 101
# evals_array = np.linspace(6000, 16000, points)

# tck = interpolate.splrep(evals_RK4_1, error_RK4_1, k = 3)
# error_RK4_1_interp = interpolate.splev(evals_array, tck)

# tck= interpolate.splrep(evals_RK4_2, error_RK4_2, k = 3)
# error_RK4_2_interp = interpolate.splev(evals_array, tck)


# plt.figure(figsize=(5,5))
# plt.plot(evals_array, error_RK4_1_interp/error_RK4_2_interp, 'r', linewidth = 2.5)
# plt.show()


# tck = interpolate.splrep(evals_RK8_1, error_RK8_1, k = 3)
# error_RK8_1_interp = interpolate.splev(evals_array, tck)

# tck= interpolate.splrep(evals_RK8_2, error_RK8_2, k = 3)
# error_RK8_2_interp = interpolate.splev(evals_array, tck)


# plt.figure(figsize=(5,5))
# plt.plot(evals_array, error_RK8_1_interp/error_RK8_2_interp, 'b', linewidth = 2.5)
# plt.show()



