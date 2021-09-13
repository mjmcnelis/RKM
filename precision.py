#!/usr/bin/env python3
import numpy as np

def precision(x):
    return np.float128(x)
    # return np.float64(x)

myfloat = type(precision(1))

