#!/usr/bin/env python3
import math
import numpy as np
from scipy.optimize import fsolve
from precision import precision
myfloat = type(precision(1))

# https://people.sc.fsu.edu/~jburkardt/py_src/backward_euler/backward_euler.py

EPS_ROOT = 1.e-4
ITER = 10


def implicit_euler(y, t, dt, y_prime, root_method = 'fixed_point', eps_root = EPS_ROOT, max_iterations = ITER):

	# stages = 1
	# dy_array = [0] * stages
	dy = 0
	dy_prev = 0

	# does fixed point iteration work for stiff problems?
	iterations = 0

	for n in range(0, max_iterations):

		if root_method is 'fixed_point':
			dy = dt * y_prime(t + dt, y + dy)
			# k1 = y_prime(t + dt, y + dt*k1)

			delta   = np.linalg.norm(dy - dy_prev, ord = None)
			dy_norm = np.linalg.norm(dy, ord = None)
			y_norm  = np.linalg.norm(y + dy, ord = None)

			iterations += 1

			# tolerance = eps_root * max(y_norm, dy_norm)
			# tolerance = eps_root * max(1, dy_norm)
			tolerance = eps_root * dy_norm

			if delta <= tolerance:
				break

			dy_prev = dy

		elif root_method is 'fsolve':
			quit()

	return y + dy, iterations



def implicit_midpoint(y, t, dt, y_prime, root_method = 'fixed_point', eps_root = EPS_ROOT, max_iterations = ITER):

	dy = 0
	dy_prev = 0

	iterations = 0

	for n in range(0, max_iterations):

		if root_method is 'fixed_point':

			dy = dt * y_prime(t + dt/2, y + dy/2)
			iterations += 1

			delta   = np.linalg.norm(dy - dy_prev, ord = None)
			dy_norm = np.linalg.norm(dy, ord = None)
			tolerance = eps_root * dy_norm

			if delta <= tolerance:
				# print('succeed = ', n)
				break

			dy_prev = dy

		elif root_method is 'fsolve':
			quit()

	if n == max_iterations - 1:
		print('fail')

	return y + dy, iterations



def standard_DIRK_step(y, t, dt, y_prime, butcher, embedded = False, root_method = 'fixed_point', eps_root = EPS_ROOT, max_iterations = ITER):

	# so far the code only supports diagonal implicit

	if embedded:
		stages = butcher.shape[0] - 2
	else:
		stages = butcher.shape[0] - 1

	dy_array = [0] * stages

	iterations = 0

	# may have to rewrite standard_RK to account for implicit stages

	# make a function called evaluate_stage(), put inside for loop

	for i in range(0, stages):

		dy_prev = 0

		# todo: if the row is not implicit, then just evaluate dy_array[i] as explicit (helps for crank nicolson)
		#		need to analyze the row

		for n in range(0, max_iterations):

			if root_method is 'fixed_point':
				dy = 0

				for j in range(0, stages):
					dy += dy_array[j] * butcher[i, j+1]

				dy_array[i] = dt * y_prime(t + dt*butcher[i,0], y + dy)

				iterations += 1

				delta   = np.linalg.norm(dy_array[i] - dy_prev, ord = None)
				dy_norm = np.linalg.norm(dy_array[i], ord = None)

				tolerance = eps_root * dy_norm

				if delta <= tolerance:
					break

				dy_prev = dy_array[i]
			else:
				print('standard_DIRK_step error: nothing yet')
				quit()

	dy = 0

	for j in range(0, stages):
		dy += dy_array[j] * butcher[stages, j+1]

	return y + dy, iterations




