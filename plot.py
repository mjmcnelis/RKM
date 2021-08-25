#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from explicit_runge_kutta import dt_MIN, dt_MAX



def plot_test(y, t, dt, t0, tf, method, adaptive, rejection_rate, function_evaluations, plot_variables, error, log_plot = False):

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), squeeze = False)

	y = y[:,0:min(plot_variables, y.shape[1])]

	# plot y
	axes[0][0].plot(t, y[:,0], 'red', label = method, linewidth = 1.5)

	axes[0][0].text(t0 + 0.1*(tf-t0), 0.2*abs(np.amax(y)) + np.amax(y), r'$\mathcal{E}^{(\mathrm{avg})}_{\mathrm{abs}} = %.2e$' % error, fontsize=10)

	for i in range(1, min(plot_variables, y.shape[1])):
		axes[0][0].plot(t, y[:,i], linewidth = 1.5)

	axes[0][0].set_ylabel('y', fontsize=12)
	axes[0][0].set_xlabel('t', fontsize=12)
	axes[0][0].set_xlim(t0, tf)
	axes[0][0].set_ylim(min(-0.01*abs(np.amax(y)), -0.1*abs(np.amin(y))) + np.amin(y), 0.5*abs(np.amax(y)) + np.amax(y))

	if log_plot:
		axes[0][0].set_yscale('log')

	axes[0][0].axhline(0, color = 'black', linewidth = 0.3)
	axes[0][0].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
	axes[0][0].tick_params(labelsize = 10)
	axes[0][0].legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)

	# plot dt
	axes[0][1].plot(t, dt, 'black', label = adaptive, linewidth = 1.5)

	if adaptive not in [None, 'RKM']:
		axes[0][1].text(t0 + 0.45*(tf-t0), 1.1*dt_MAX, 'R = %.1f%%' % rejection_rate, fontsize = 10)

	axes[0][1].text(t0 + 0.05*(tf-t0), 1.15*dt_MAX, 'steps = %d' % (function_evaluations/4), fontsize = 10)
	axes[0][1].text(t0 + 0.05*(tf-t0), 1.05*dt_MAX, 'FE = %d' % function_evaluations, fontsize = 10)

	axes[0][1].set_ylabel(r'${\Delta t}_n$', fontsize=12)
	axes[0][1].set_xlabel('t', fontsize=12)
	axes[0][1].set_xlim(t0, tf)

	axes[0][1].set_ylim(0, 1.25*dt_MAX)
	axes[0][1].axhline(dt_MIN, color = 'black', linewidth = 0.3)
	axes[0][1].axhline(dt_MAX, color = 'black', linewidth = 0.3)

	axes[0][1].set_xticks(np.round(np.linspace(t0, tf, 5), 2))
	axes[0][1].legend(fontsize = 10, borderpad = 1, labelspacing = 0, handlelength = 2, handletextpad = 1, frameon = False)
	axes[0][1].tick_params(labelsize = 10)

	fig.tight_layout(pad = 2)
	# fig.suptitle(r"$y^{''} = -10 - \frac{1}{2}y'$", fontsize = 12)
	fig.savefig("test_plot.png", dpi = 200)
	plt.show()




