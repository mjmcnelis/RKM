#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt

import exact_solution
from exact_solution import t0, tf, solution, y_exact, y_prime, jacobian
from ode_solver import ode_solver
from explicit_runge_kutta import dt_MAX
import test_parameters											# for evolve.py, use a parameter file instead

y0 = y_exact(t0)												# set initial conditions

adaptive       = test_parameters.adaptive						# set parameters
method         = test_parameters.method
norm           = test_parameters.norm
dt0            = test_parameters.dt0
eps            = test_parameters.eps
root           = test_parameters.root
error_type     = test_parameters.error_type
average        = test_parameters.average
interpolate    = test_parameters.interpolate
plot_variables = test_parameters.plot_variables


# evolve system
y, t, dt, evaluations, reject = ode_solver(y0, t0, tf, dt0, y_prime, method, adaptive = adaptive, jacobian = jacobian, norm = norm, eps = eps, root = root)


# compute numerical error
error = exact_solution.compute_error_of_exact_solution(t, y, y_exact, error_type = error_type, average = average)


# plot results
#---------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), squeeze = False)

y = y[:,0:min(plot_variables, y.shape[1])]


# plot y
axes[0][0].plot(t, y[:,0], 'red', label = method, linewidth = 1.5)

if average:
	if error_type == 'relative':
		axes[0][0].text(t0 + 0.1*(tf-t0), 0.2*abs(np.amax(y)) + np.amax(y), r'$\mathcal{E}^{(\mathrm{avg})}_{\mathrm{rel}} = %.2e$' % error, fontsize=10)
	else:
		axes[0][0].text(t0 + 0.1*(tf-t0), 0.2*abs(np.amax(y)) + np.amax(y), r'$\mathcal{E}^{(\mathrm{avg})}_{\mathrm{abs}} = %.2e$' % error, fontsize=10)
else:
	if error_type == 'relative':
		axes[0][0].text(t0 + 0.1*(tf-t0), 0.2*abs(np.amax(y)) + np.amax(y), r'$\mathcal{E}^{(\mathrm{max})}_{\mathrm{rel}} = %.2e$' % error, fontsize=10)
	else:
		axes[0][0].text(t0 + 0.1*(tf-t0), 0.2*abs(np.amax(y)) + np.amax(y), r'$\mathcal{E}^{(\mathrm{max})}_{\mathrm{abs}} = %.2e$' % error, fontsize=10)

for i in range(1, min(plot_variables, y.shape[1])):
	axes[0][0].plot(t, y[:,i], linewidth = 1.5)

axes[0][0].set_ylabel('y', fontsize=12)
axes[0][0].set_xlabel('t', fontsize=12)
axes[0][0].set_xlim(t0, tf)
axes[0][0].set_ylim(min(-0.01*abs(np.amax(y)), -0.1*abs(np.amin(y))) + np.amin(y), 0.5*abs(np.amax(y)) + np.amax(y))

if solution in ['exponential', 'inverse_power']:
	axes[0][0].set_yscale('log')

axes[0][0].axhline(0, color = 'black', linewidth = 0.3)
axes[0][0].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][0].tick_params(labelsize = 10)
axes[0][0].legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)


# plot dt
axes[0][1].plot(t, dt, 'black', label = adaptive, linewidth = 1.5)
axes[0][1].text(t0 + 0.45*(tf-t0), 1.1*max(dt), 'R = %.1f%%' % reject, fontsize = 10)
axes[0][1].text(t0 + 0.05*(tf-t0), 1.1*max(dt), 'FE = %d' % evaluations, fontsize = 10)
axes[0][1].set_ylabel(r'${\Delta t}_n$', fontsize=12)
axes[0][1].set_xlabel('t', fontsize=12)
axes[0][1].set_xlim(t0, tf)
axes[0][1].set_ylim(0.8*min(dt), 1.25*max(dt))
axes[0][1].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][1].legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
axes[0][1].tick_params(labelsize = 10)

fig.tight_layout(pad = 2)
plt.show()
#---------------------------------------------------------------------------------------




