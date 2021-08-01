#!/usr/bin/env python3
import math
import numpy as np
from numpy.core.numeric import identity
from scipy.optimize import fsolve
from precision import precision
myfloat = type(precision(1))

# https://people.sc.fsu.edu/~jburkardt/py_src/backward_euler/backward_euler.py

EPS_ROOT = 1.e-5
ITER = 10

def standard_DIRK_step(y, t, dt, y_prime, jacobian, butcher, explicit_stages, embedded = False, 
					   root = 'fixed_point', eps_root = EPS_ROOT, max_iterations = ITER):

	# should I default jacobian to 0?

	# todo: i should pass standard_RK or standard_DIRK, etc (with non-common parameters as kargs)

	# so far, code supports diagonal implicit Runge Kutta
	# todo: in standard_RK, account for implicit stages (make function called evaluate_stage(), put inside for loop?)

	# print(y_prime(t, y), y_prime(t, y).shape)
	# print()
	# print(jacobian(t, y), jacobian(t, y).shape)

	if embedded:
		stages = butcher.shape[0] - 2
	else:
		stages = butcher.shape[0] - 1

	dimension = y.shape[0]
	identity = np.identity(dimension)								

	dy_array = [0] * stages

	iterations = 0	

	for i in range(0, stages):
		
		if explicit_stages[i]:										# if stage is explicit, can evaluate it directly
			dy = 0

			for j in range(0, i):
				dy += dy_array[j] * butcher[i, j+1]

			dy_array[i] = dt * y_prime(t + dt*butcher[i,0], y + dy)

			iterations += 1
		
		else:														# otherwise, need to iterate stage until convergence is reached
			ci  = butcher[i,0]
			aii = butcher[i, i+1]

			dy = 0

			for j in range(0, i):
				dy += dy_array[j] * butcher[i, j+1]   				# add previously known stages (fixed during iteration)

			z = dy_array[i]											# current stage dy^(i) that we need to iterate
			z_prev = 0

			if root is 'newton_fast':
				J = identity  -  aii * dt * jacobian(t, y)			# evaluate jacobian once

				iterations += dimension

			for n in range(0, max_iterations):						# start iteration loop

				if root is not 'fixed_point':

					g = z - dt*y_prime(t + dt*ci, y + dy + z*aii)	# solve nonlinear system g(z) = 0 via Newton's method

					iterations += 1

					if root is 'newton':							# evaluate jacobian for every iteration
						J = identity  -  aii * dt * jacobian(t + dt*ci, y + dy + z*aii)

						iterations += dimension

					dz = np.linalg.solve(J.astype('float64'), -g.astype('float64'))		

					z += dz											# Newton iteration (linalg only supports float64)

				else:
					z = dt * y_prime(t + dt*ci, y + dy + z*aii)		# solve nonlinear system g(z) = 0 via fixed point iteration

					iterations += 1

				delta   = np.linalg.norm(z - z_prev, ord = None)
				dy_norm = np.linalg.norm(z, ord = None)

				tolerance = eps_root * dy_norm

				if delta <= tolerance:								# check for convergence of solution
					break
					
				z_prev = z											# get previous value for next iteration

			dy_array[i] = z

	dy = 0

	for j in range(0, stages):
		dy += dy_array[j] * butcher[stages, j+1]					# evaluate RK update

	return y + dy, iterations




