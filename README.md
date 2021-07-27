**RKM (c) Mike McNelis**

Created on 4/20/2021\
Last edited on 7/27/2021

## Summary
A Runge-Kutta ODE solver with a new adaptive step size method.

This Python code evolves ordinary differential equations with one of three adaptive methods:

| ||
|:----:|:-------------:|
| RKM  | new method    |
| ERK  | embedded      |
| SDRK | step doubling |

The RKM algorithm was first used in one of my papers to solve fluid dynamic equations (see Sec. 3.6 for the derivation)

    M. McNelis, D. Bazow and U. Heinz, arXiv:2101.02827 (2021)

This repository is currently under development, but you can run `evolve.py` or `test_exact.py`to generate an example plot.


## Runge-Kutta methods

A list of Runge-Kutta methods in Butcher tableau notation can be found in `butcher_tables`. They are organized in two categories: `standard` for RKM and SDRK, and `embedded` for ERK.

Note: we do not consider multiple secondary pairs, if any, in the embedded schemes (e.g. Bogacki-Shampine 5(4) does not include the second fourth-order pair).


## Status

We have started adding new features, such as implicit Runge-Kutta solvers and interpolators for dense output. 



