#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt



def compute_dt(dt_prev, y, y_prev, k1, eps, p, l, low = 0.2, high = 1.5):

    eps = eps**(1/p)
    high = high**(1/p)

    y_star = y + dt_prev*k1

    C = 2 * (y_star - 2*y + y_prev) / dt_prev**2

    C_norm = np.linalg.norm(C,  ord = l)
    y_norm = np.linalg.norm(y,  ord = l)
    f_norm = np.linalg.norm(k1, ord = l)

    if C_norm == 0:
        dt = dt_prev
    else:
        if (C_norm * y_norm) > (2 * eps * f_norm**2):
            dt = (2 * eps * y_norm / C_norm)**0.5
        else:
            dt = 2 * eps * f_norm / C_norm

        dt = min(high*dt_prev, max(low*dt_prev, dt))

    return dt


def f(t, y):
    return y


def RK3_evolve(y0, t0, tf, dt0, f, adaptive = True, eps = 1.e-8, l = None):

    y_array  = np.empty(shape=[0])
    t_array  = np.empty(shape=[0])
    p = 3

    y = y0
    t = t0
    dt = dt0

    for n in range(0, 10000):
        y_array = np.append(y_array, y)
        t_array = np.append(t_array, t)

        k1 = f(t, y)

        if (n > 0 and adaptive):
            dt = compute_dt(dt, y, y_prev, k1, eps, p, l)

        k2 = f(t + dt,   y + dt*k1)
        k3 = f(t + dt/2, y + dt*k1/4 + dt*k2/4)

        y_prev = y

        y += dt * (k1/6 + k2/6 + 2*k3/3)

        if t >= tf:
            break

        t += dt

    return y_array, t_array


t0 = -2
tf = 2
dt0 = 0.01
y0 = math.exp(t0)

y, t = RK3_evolve(y0, t0, tf, dt0, f)

y_exact = [0] * len(t)

for i in range(0, len(t)):
    y_exact[i] = math.exp(t[i])

plt.plot(t, y_exact, 'black', label = 'RK4M', linewidth = 1.5)
plt.plot(t, y, 'r--', label = 'RK4M', linewidth = 1.5)
plt.ylabel('y', fontsize=14)
plt.xlabel('t', fontsize=14)
plt.show()

