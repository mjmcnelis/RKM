#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

from ode_solver import ode_solver
import butcher_table
import precision
import exact_solution                           # for solution

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

solution = exact_solution.solution

if solution is 'sine':                          # set initial conditions
    A = exact_solution.A                        # can bring back precision()
    cycles = exact_solution.cycles
    t0 = 0
    tf = 2*math.pi/A * cycles
    norm = 1
else:
    t0 = -10
    tf = 10
    norm = None

y0 = exact_solution.y_exact(t0, solution)       # todo: replace with own solution
dt0 = 0.01

solver = 'RKM'

if solver is 'RKM' or solver is 'SDRK':
    method = 'dormand_prince_8'
else:
    method = 'dormand_prince_8_7'

y, t, dt, evaluations, finish = ode_solver(y0, t0, tf, dt0, solver, method, norm = norm)

error = exact_solution.compute_error_of_exact_solution(t, y, solution, error_type = 'relative', norm = norm)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), squeeze = False)

axes[0][0].plot(t, y[:,0], 'blue', label = butcher_table.methods_dict[method], linewidth = 1.5)
axes[0][0].text(t0 + 0.15*(tf-t0), 0.95*max(y[:,0]), r'$\langle \mathcal{E}_{\mathrm{rel}} \rangle = %.2e$' % error, fontsize=10)
axes[0][0].set_ylabel('y', fontsize=14)
axes[0][0].set_xlabel('t', fontsize=14)
axes[0][0].set_xlim(t0, tf)
# axes[0][0].set_yscale('log')
axes[0][0].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][0].legend(fontsize = 14, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
axes[0][0].tick_params(labelsize = 10)

axes[0][1].plot(t, dt, 'blue', linestyle = 'dashed', linewidth = 1.5)
axes[0][1].set_ylabel(r'${\Delta t}_n$', fontsize=14)
axes[0][1].set_xlabel('t', fontsize=14)
axes[0][1].set_xlim(t0, tf)
axes[0][1].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][1].tick_params(labelsize = 10)

fig.tight_layout()
fig.savefig("test_plot.png", dpi = 200)




