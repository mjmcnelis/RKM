#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from ode_solver import ode_solver
from explicit_runge_kutta import dt_MAX
from precision import precision
import exact_solution
from exact_solution import solution, A, cycles

myfloat = type(precision(1))

if solution is 'sine':
	t0 = 0
	tf = 2*math.pi/A * cycles
elif solution is 'inverse_power':
	t0 = 0.0001
	tf = 10
else:
	t0 = -10
	tf = 10
y0 = exact_solution.y_exact(t0)
y_prime = exact_solution.y_prime

dt0 = dt_MAX
adaptive = 'RKM'

if adaptive is not 'ERK':
	method = 'RK4'

	if adaptive is 'RKM':
		suffix = 'M'
	else:
		suffix = 'SD'
else:
	method = 'F45'
	suffix = ''

y, t, dt, evaluations, reject, finish = ode_solver(y0, t0, tf, dt0, y_prime, adaptive, method)

print('rejection rate = %.1f %%' % reject)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), squeeze = False)
axes[0][0].plot(t, y[:,0], 'blue', label = method + suffix, linewidth = 1.5)

y_exact = exact_solution.y_exact
error = exact_solution.compute_error_of_exact_solution(t, y, y_exact, error_type = 'relative')

axes[0][0].text(t0 + 0.1*(tf-t0), 0.9*max(y[:,0]), r'$\langle \mathcal{E}_{\mathrm{rel}} \rangle = %.2e$' % error, fontsize=10)

axes[0][0].set_ylabel('y', fontsize=12)
axes[0][0].set_xlabel('t', fontsize=12)
axes[0][0].set_xlim(t0, tf)
# axes[0][0].set_yscale('log')
axes[0][0].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][0].legend(fontsize = 12, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
axes[0][0].tick_params(labelsize = 10)

axes[0][1].plot(t, dt, 'blue', linestyle = 'dashed', linewidth = 1.5)
axes[0][1].text(t0 + 0.1*(tf-t0), 1.1*max(dt), 'FE = %d' % evaluations, fontsize = 10)
axes[0][1].set_ylabel(r'${\Delta t}_n$', fontsize=12)
axes[0][1].set_xlabel('t', fontsize=12)
axes[0][1].set_xlim(t0, tf)
axes[0][1].set_ylim(0.8*min(dt), 1.2*max(dt))
axes[0][1].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][1].tick_params(labelsize = 10)
fig.tight_layout()
# fig.savefig("test_explicit.png", dpi = 200)
plt.show()




