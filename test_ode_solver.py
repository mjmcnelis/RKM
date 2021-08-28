#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

from parameters import parameters
from exact_solution import t0, tf, solution, y_exact, y_prime, jacobian, compute_error_of_exact_solution
from ode_solver import ode_solver
from plot import plot_test

# set initial conditions
y0 = y_exact(t0)


# evolve ode system
y, t, dt, evaluations, reject = ode_solver(y0, t0, tf, y_prime, parameters, jacobian = jacobian)


# evaluate numerical accuracy
error = compute_error_of_exact_solution(t, y, y_exact)


# plot solution
log_plot = solution in ['exponential', 'inverse_power']

adaptive       = parameters['adaptive']
method         = parameters['method']
plot_variables = 2

plot_test(y, t, dt, t0, tf, method, adaptive, reject, evaluations, plot_variables, error, log_plot = log_plot)




