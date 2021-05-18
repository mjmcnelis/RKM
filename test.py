#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

from ode_solver import ode_solver
import precision
import exact_solution                           # for solution

solution = exact_solution.solution

if solution is 'sine':                          # set initial conditions
    A = exact_solution.A                        # can bring back precision()
    t0 = 0
    tf = math.pi/A * 5
else:
    t0 = -10
    tf = 10

y0 = exact_solution.y_exact(t0, solution)       # todo: replace with own solution
dt0 = 0.001


solver = 'ERK'
method = 'dormand_prince_8_7'

y, t, dt, evaluations, finish = ode_solver(y0, t0, tf, dt0, solver, method)

plt.figure(figsize = (5,4))
plt.plot(t, y[:,0], 'r', linewidth = 1.5)
plt.plot(t, dt, 'b:', linewidth = 1.5)
plt.savefig("test.png", dpi = 200)

error = exact_solution.compute_error_of_exact_solution(t, y, solution)

print('average absolute error =', error)




