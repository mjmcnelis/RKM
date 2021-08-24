#!/usr/bin/env python3
import math
import numpy as np
from numpy.core.numeric import identity
from scipy.optimize import fsolve
from precision import precision
from explicit_runge_kutta import dt_MIN, dt_MAX, LOW, HIGH, HIGH_RKM	# todo: make a separate parameters file
myfloat = type(precision(1))

EPS_ROOT = 1.e-6
ITERATIONS = 2

# EPS_ROOT = 1.e-3
# ITERATIONS = 1

RECOMPUTE_K1 = False

# compute dt using RKM algorithm
def compute_dt_RKM(dt_prev, y, y_prev, k1, method, eps = 1.e-8, norm = None,
				   dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH_RKM):

	# same subroutine in RKM_step()

	order = int(method.split('_')[-1])

	high = high**(1/order)

	y_star = y + dt_prev*k1

	C = 2 * (y_star - 2*y + y_prev) / dt_prev**2

	C_norm = np.linalg.norm(C,  ord = norm)
	y_norm = np.linalg.norm(y,  ord = norm)
	f_norm = np.linalg.norm(k1, ord = norm)

	if C_norm == 0:
		dt = dt_prev
	else:
		if (C_norm * y_norm) > (2 * eps * f_norm**2):
			dt = (2 * eps * y_norm / C_norm)**0.5
		else:
			dt = 2 * eps * f_norm / C_norm

		dt = min(high*dt_prev, max(low*dt_prev, dt))

	dt = min(dt_max, max(dt_min, dt))

	return dt



# diagonal implicit Runge Kutta step
def DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, embedded = False,
				  root = 'newton', eps_root = EPS_ROOT, max_iterations = ITERATIONS,
				  adaptive = None, method = None, y_prev = None, eps = 1.e-8, norm = None):


	# I feel like now there's a bug with adaptive = None (does fixed time step skip any calculations, evaluations?)


	# last five arguments are for RKM

	# RKM note: should I always use forward Euler stage f(t,y) for RKM algorithm
	#			because I probably need to recompute the first implicit stage anyway (since it depends on dt)

	# RKM idea: if first stage is not explicit, then should I save new dt for the next Newton iteration?

	# print(y_prime(t, y), y_prime(t, y).shape)
	# print()
	# print(jacobian(t, y), jacobian(t, y).shape)

	if embedded:
		c = butcher[:-2, 0]
		A = butcher[:-2, 1:]
		b = butcher[-2, 1:]
		bhat = butcher[-1, 1:]
		stages = butcher.shape[0] - 2
	else:
		c = butcher[:-1, 0]
		A = butcher[:-1, 1:]
		b = butcher[-1, 1:]
		stages = butcher.shape[0] - 1

	dimension = y.shape[0]
	identity = np.identity(dimension)

	dy_array = [0] * stages

	evaluations = 0

	for i in range(0, stages):

		if stage_explicit[i]:										# if stage is explicit, can evaluate it directly (no iterations required)
			if i == 0 and adaptive is 'RKM':						# if first stage is explicit, then c[0] = dy = 0
				k1 = y_prime(t, y)

				dt = compute_dt_RKM(dt, y, y_prev, k1, method, eps = eps, norm = norm)

				dy_array[i] = dt * k1

			else:
				dy = 0

				for j in range(0, i):
					dy += dy_array[j] * A[i,j]

				dy_array[i] = dt * y_prime(t + dt*c[i], y + dy)

			evaluations += 1

		else:														# otherwise, need to iterate stage until convergence is reached
			Aii = A[i,i]
			dy = 0

			for j in range(0, i):
				dy += dy_array[j] * A[i,j]   						# add previously known stages (fixed during iteration)

			z = dy_array[i]											# current stage dy^(i) that we need to iterate
			z_prev = 0

			if i == 0 and adaptive is 'RKM':
				k1 = y_prime(t + dt*c[i], y)

				evaluations += 1

				dt = compute_dt_RKM(dt, y, y_prev, k1, method, eps = eps, norm = norm)

				if RECOMPUTE_K1:
					k1 = y_prime(t + dt*c[i], y)				     # should I re-evaluate it?
					evaluations += 1                                 # I don't see much impact (just more evals)

				if root is not 'fixed_point':
					g = - dt * k1
				else:
					z = dt * k1

			if root is 'newton':
				J = identity  -  Aii * dt * jacobian(t, y)			# only evaluate jacobian once

				evaluations += dimension

			for n in range(0, max_iterations):						# start iteration loop

				if root is not 'fixed_point':

					if adaptive is None or n > 0 or i > 0:		# solve nonlinear system g(z) = 0 via Newton's method
						g = z - dt*y_prime(t + dt*c[i], y + dy + z*Aii)

						evaluations += 1

					if root is 'newton_full':						# evaluate jacobian for every iteration
						J = identity  -  Aii * dt * jacobian(t + dt*c[i], y + dy + z*Aii)

						evaluations += dimension

					dz = np.linalg.solve(J.astype('float64'), -g.astype('float64'))

					z += dz											# Newton iteration (linalg only supports float64)

					delta = np.linalg.norm(g, ord = norm)
					tolerance = eps_root

					# should I also consider dz = z - z_prev being stuck, i.e. smaller than g(z)?

					# like, if either g(z) ~ 0 or dz ~ 0.01*z then break

					if delta <= tolerance:							# check for convergence of solution g(z) = 0
						break

					# maybe include this
					# z_prev = z.copy()

				else:
					if adaptive is None or n > 0 or i > 0:		# solve nonlinear system g(z) = 0 via fixed point iteration
						z = dt * y_prime(t + dt*c[i], y + dy + z*Aii)

						evaluations += 1

						delta   = np.linalg.norm(z - z_prev, ord = norm)
						dy_norm = np.linalg.norm(z, ord = norm)

						tolerance = eps_root * dy_norm

						if delta <= tolerance:						# check for convergence of solution dz = 0
							break

						z_prev = z.copy()							# get previous value for next iteration

			dy_array[i] = z

	dy = 0

	for j in range(0, stages):										# primary RK update
		dy += dy_array[j] * b[j]

	if adaptive is 'RKM':
		return y + dy, y, dt, evaluations							# return variables for RKM step

	elif embedded:
		dyhat = 0

		for j in range(0, stages):									# secondary RK update (for embedded schemes)
			dyhat += dy_array[j] * bhat[j]

		return (y + dyhat), (y + dy), evaluations					# updated solution (secondary, primary)

	return y + dy, evaluations										# updated solution (primary)



# fully implicit Runge Kutta step
def FIRK_standard(y, t, dt, y_prime, jacobian, butcher, embedded = False,
				  root = 'fixed_point', eps_root = EPS_ROOT, max_iterations = ITERATIONS):

	if embedded:
		c = butcher[:-2, 0]
		A = butcher[:-2, 1:]  										# get c_i, A_ij, b_i and stages
		b = butcher[-2, 1:]
		bhat = butcher[-1, 1:]
		stages = butcher.shape[0] - 2
	else:
		c = butcher[:-1, 0]
		A = butcher[:-1, 1:]
		b = butcher[-1, 1:]
		stages = butcher.shape[0] - 1

	dimension = y.shape[0]
	identity = np.identity(dimension * stages)

	dy_array = [0] * stages

	iterations = 0

	J = identity  -  dt * (A * jacobian(t,y)).reshape(identity.shape)

	# print(A * jacobian(t,y).reshape(identity.shape))

	# z = np.zeros((stages,1))								# stages dy^(i) that we need to iterate
	z = np.zeros(stages)
	z_prev = 0

	for n in range(0, max_iterations):

		for i in range(0, stages):
			dy = 0

			for j in range(0, stages):
				dy += dy_array[j] * A[i,j]

			dy_array[i] = dt * y_prime(t + dt*c[i], y + dy)

			z[i] = dy_array[i]


		z_reshape = z.reshape(dimension*stages, 1)

	# 	g = z - dt*y_prime(t + dt*ci, y + dy + z*aii)			# could i move this in the inner ij-loop?

		print(z, z.shape)
		quit()

	# 	full jacobian would go here (or be iterated in the ij loop)

		# dz = np.linalg.solve(J.astype('float64'), -g.astype('float64'))
		# z_reshape += dz											# Newton iteration (linalg only supports float64)


	# 	delta   = np.linalg.norm(z - z_prev, ord = None)
	# 	dy_norm = np.linalg.norm(z, ord = None)

	# 	tolerance = eps_root * dy_norm

	# 	if delta <= tolerance:								# check for convergence of solution
	# 		break

	# 	z_prev = z											# get previous value for next iteration

	# 	dy_array[i] = z

	dy = 0

	for j in range(0, stages):
		dy += dy_array[j] * b[j]									# evaluate RK update

	return y + dy, iterations



# step doubling DIRK step
def SDDIRK_step(y, t, dt, y_prime, jacobian, method, butcher, stage_explicit, root = 'fixed_point', eps = 1.e-8,
 			    norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH, S = 0.9, max_attempts = 100):

	# routine is very similar to SDRK_step()

	order = int(method.split('_')[-1])                      # get order of method
	power = 1 / (1 + order)

	high = high**(order/(1+order))

	evaluations = 0
	rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

	for i in range(0, max_attempts):

		dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        # if (dt == dt_min or dt == dt_max) and rescale < 1:
        #     print('SDDIRK_step flag: dt = %.2e at t = %.2f (change dt_min, dt_max)' % (dt, t))

        # full RK step
		y1, evals_1 = DIRK_standard(y, t, dt, y_prime, jacobian, butcher, stage_explicit, root = root, norm = norm)

        # two half RK steps
		y_mid, evals_mid = DIRK_standard(y, t, dt/2, y_prime, jacobian, butcher, stage_explicit, root = root, norm = norm)
		t_mid = t + dt/2

		y2, evals_2 = DIRK_standard(y_mid, t_mid, dt/2, y_prime, jacobian, butcher, stage_explicit, root = root, norm = norm)

		evaluations += (evals_1 + evals_mid + evals_2)

		error = (y2 - y1) / (2**order - 1)                  # estimate local truncation error
		yR = y2 + error                                     # propose updated solution (Richardson extrapolation)

		error_norm = np.linalg.norm(error, ord = norm)      # error norm
		y_norm = np.linalg.norm(yR, ord = norm)
		dy_norm = np.linalg.norm(yR - y, ord = norm)

		tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

		if error_norm == 0:
			rescale = 1                                     # prevent division by 0
		else:
			rescale = (tolerance / error_norm)**power       # scaling factor
			rescale = min(high, max(low, S*rescale))        # control rate of change

		if error_norm <= tolerance:                         # check if attempt succeeded
			dt_next = min(dt_max, max(dt_min, dt*rescale))  # impose dt_min <= dt <= dt_max

            # if dt_next == dt_min or dt_next == dt_max:
            #     print('SDDIRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

			return yR, dt, dt_next, i + 1, evaluations      # updated solution, current dt, next dt, number of attempts, function evaluations

		else:
			rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

	dt_next = min(dt_max, max(dt_min, dt*rescale))

    # if dt_next == dt_min or dt_next == dt_max:
    #     print('SDDIRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

	return yR, dt, dt_next, i + 1, evaluations              # return last attempt



# embedded DIRK step
def EDIRK_step(y0, t, dt, y_prime, jacobian, method, butcher, stage_explicit, root = 'fixed_point', eps = 1.e-8,
 			   norm = None, dt_min = dt_MIN, dt_max = dt_MAX, low = LOW, high = HIGH, S = 0.9, max_attempts = 100):

	# routine is very similar to ERK_step()

	order = int(method.split('_')[-2])                      # order of primary method
	order_hat = int(method.split('_')[-1])                  # order of secondary method

	order_max = max(order, order_hat)
	order_min = min(order, order_hat)
	power = 1 / (1 + order_min)

	evaluations = 0
	rescale = 1                                             # for scaling dt <- dt*rescale (starting value = 1)

	for i in range(0, max_attempts):

		dt = min(dt_max, max(dt_min, dt*rescale))           # decrease step size for next attempt

        # if (dt == dt_min or dt == dt_max) and rescale < 1:
        #     print('EDIRK_step flag: dt = %.2e at t = %.2f (change dt_min, dt_max)' % (dt, t))

		# propose updated solution (secondary, primary)
		yhat, y, evals = DIRK_standard(y0, t, dt, y_prime, jacobian, butcher, stage_explicit, embedded = True, root = root, norm = norm)

		evaluations += evals

		error_norm = np.linalg.norm(y - yhat, ord = norm)   # error norm
		y_norm = np.linalg.norm(y, ord = norm)
		dy_norm = np.linalg.norm(y - y0, ord = norm)

		tolerance = eps * max(y_norm, dy_norm)              # compute tolerance

		if error_norm == 0:
			rescale = 1                                     # prevent division by 0
		else:
			rescale = (tolerance / error_norm)**power       # scaling factor
			rescale = min(high, max(low, S*rescale))        # control rate of change

		if error_norm <= tolerance:                         # check if attempt succeeded
			dt_next = min(dt_max, max(dt_min, dt*rescale))  # impose dt_min <= dt <= dt_max

            # if dt_next == dt_min or dt_next == dt_max:
            #     print('EDIRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

			return y, dt, dt_next, i + 1, evaluations       # updated solution, current dt, next dt, number of attempts, function evaluations

		else:
			rescale = min(S, rescale)                       # enforce rescale < 1 if attempt failed

	dt_next = min(dt_max, max(dt_min, dt*rescale))

    # if dt_next == dt_min or dt_next == dt_max:
    #     print('EDIRK_step flag: dt_next = %.2e at t = %.2f (change dt_min, dt_max)' % (dt_next, t))

	return y, dt, dt_next, i + 1, evaluations               # return last attempt