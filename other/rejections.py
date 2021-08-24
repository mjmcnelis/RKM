#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy import interpolate

points = 101
evals_array = np.linspace(4000, 100000, points)

efficiency_RK4_1 = np.loadtxt('efficiency_RK4_E_ideal.dat')
efficiency_RK4_2 = np.loadtxt('efficiency_RK4_E.dat')

error_RK4_1, evals_RK4_1 = efficiency_RK4_1[:,0], efficiency_RK4_1[:,1]
error_RK4_2, evals_RK4_2 = efficiency_RK4_2[:,0], efficiency_RK4_2[:,1]

tck = interpolate.splrep(evals_RK4_1, error_RK4_1, k = 3)
error_RK4_1_interp = interpolate.splev(evals_array, tck)

tck= interpolate.splrep(evals_RK4_2, error_RK4_2, k = 3)
error_RK4_2_interp = interpolate.splev(evals_array, tck)

plt.plot(evals_array, error_RK4_1_interp/error_RK4_2_interp, 'r', linewidth = 2.5)
plt.show()

# plt.plot(evals_array, error_RK4_1_interp, 'b', linewidth = 2.5)
# plt.plot(evals_array, error_RK4_2_interp, 'b', linewidth = 2.5)

# plt.plot(evals_RK4_1, error_RK4_1, 'k--', linewidth = 2.5)
# plt.plot(evals_RK4_2, error_RK4_2, 'k--', linewidth = 2.5)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()


