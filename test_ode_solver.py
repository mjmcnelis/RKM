#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

import test_parameters											# for evolve.py, use a parameter file instead
import exact_solution
from exact_solution import t0, tf, solution, y_exact, y_prime, jacobian
from ode_solver import ode_solver
from plot import plot_test

y0 = y_exact(t0)												# set initial conditions

adaptive       = test_parameters.adaptive						# test parameters
method         = test_parameters.method
norm           = test_parameters.norm
dt0            = test_parameters.dt0
eps            = test_parameters.eps
root           = test_parameters.root
error_type     = test_parameters.error_type
average        = test_parameters.average
interpolate    = test_parameters.interpolate					# need to work on interpolation
plot_variables = test_parameters.plot_variables

# evolve ode system
y, t, dt, evaluations, reject = ode_solver(y0, t0, tf, dt0, y_prime, method, adaptive = adaptive, jacobian = jacobian, norm = norm, eps = eps, root = root)

# evaluate numerical accuracy
error = exact_solution.compute_error_of_exact_solution(t, y, y_exact, error_type = error_type, average = average)

log_plot = solution in ['exponential', 'inverse_power']

plot_test(y, t, dt, t0, tf, method, adaptive, reject, evaluations, plot_variables, error, average, error_type, log_plot = log_plot)




