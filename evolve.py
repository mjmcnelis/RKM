#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import ode_solver as ode
import interpolation as interp
from explicit_runge_kutta import dt_MAX
from precision import precision
myfloat = type(precision(1))

#------------------------------------------------------------
def f(t, y):
    # return -2 * t * y
    return np.array([y[1], - 10000*y[0]], dtype = myfloat).reshape(-1,1)

def y_initial(t):
	# return np.array([math.exp(-t*t)], dtype = myfloat).reshape(-1,1)
	return np.array([math.sin(100*t), 100*math.cos(100*t)], dtype = myfloat).reshape(-1,1)


# todo: load from parameters file

t0 = 0
tf = 0.1
# t0 = 0
# tf = 0.4 * math.pi
dt0 = dt_MAX
y0 = y_initial(t0)

solver = 'RKM'
method = 'RK4'
interpolate = False			# there's a bug in interpolater right now
eps = 1.e-8
norm = None
plot_variables = 1

# todo: pass RK parameters to ode solver

y, t, dt, evaluations, reject, finish = ode.ode_solver(y0, t0, tf, dt0, f, solver, method, norm = norm, eps = eps)


# todo: interpolate multi-dimension vectors
if interpolate:
	y, t, dt = interp.interpolate_solution(y, t, dt)

#------------------------------------------------------------



# make a plot notebook for this

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), squeeze = False)

if solver == 'RKM':
	method += 'M'
elif solver == 'SDRK':
	method += 'SD'

y = y[:,0:min(plot_variables, y.shape[1])]

axes[0][0].plot(t, y[:,0], 'red', label = method, linewidth = 1.5)

for i in range(1, min(plot_variables, y.shape[1])):
	axes[0][0].plot(t, y[:,i], linewidth = 1.5)

axes[0][0].set_ylabel('y', fontsize=12)
axes[0][0].set_xlabel('t', fontsize=12)
axes[0][0].set_xlim(t0, tf)
axes[0][0].set_ylim(min(-0.01*abs(np.amax(y)), -0.1*abs(np.amin(y))) + np.amin(y), 0.4*abs(np.amax(y)) + np.amax(y))
# axes[0][0].set_yscale('log')
axes[0][0].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][0].legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
axes[0][0].tick_params(labelsize = 10)

axes[0][1].plot(t, dt, 'black', linewidth = 1.5)
if solver != 'RKM':
	axes[0][1].text(t0 + 0.7*(tf-t0), 1.1*max(dt), 'R = %.1f%%' % reject, fontsize = 10)
axes[0][1].text(t0 + 0.1*(tf-t0), 1.1*max(dt), 'FE = %d' % evaluations, fontsize = 10)
axes[0][1].set_ylabel(r'${\Delta t}_n$', fontsize=12)
axes[0][1].set_xlabel('t', fontsize=12)
axes[0][1].set_xlim(t0, tf)
axes[0][1].set_ylim(0.8*min(dt), 1.2*max(dt))
axes[0][1].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
axes[0][1].tick_params(labelsize = 10)
fig.tight_layout()
plt.show()





