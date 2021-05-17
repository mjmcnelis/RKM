#!/usr/bin/env python3
import math
import numpy as np
import precision

B = 5

# exact solution
def y_exact(t):
    if solution is 'logistic':
        return np.array([math.exp(t) / (1 + math.exp(t)) - 0.5], dtype = myfloat).reshape(-1,1)
    elif solution is 'gaussian':
        return np.array([math.exp(-t*t)], dtype = myfloat).reshape(-1,1)
    elif solution is 'inverse_power':
        return np.array([1/(t + 10.01)], dtype = myfloat).reshape(-1,1)
    elif solution is 'sine':
        return np.array([math.sin(B*t), B*math.cos(B*t)], dtype = myfloat).reshape(-1,1)
    else:
        return np.array([math.exp(10*t)], dtype = myfloat).reshape(-1,1) # default is exponential


# dy/dt corresponding to exact solution
def y_prime(t, y):
    if solution is 'logistic':
        return (y + 0.5) * (0.5 - y)
    elif solution is 'gaussian':
        return - 2 * t * y
    elif solution is 'inverse_power':
        return -y**2
    elif solution is 'sine':
        return np.array([y[1], - B*B*y[0]], dtype = myfloat).reshape(-1,1)
    else:
        return 10*y