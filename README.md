**RKM (c) Mike McNelis**

Created on 4/20/2021\
Last edited on 9/12/2021

## Summary
An ODE solver with a new adaptive time stepping algorithm.

This Python code can evolve ordinary differential equations with one of three adaptive methods:

|      |               |
|:----:|:-------------:|
| RKM  | new method    |
| ERK  | embedded      |
| SDRK | step doubling |

The RKM algorithm was first used in one of my papers to solve fluid dynamic equations (see Sec. 3.6 for the derivation)

    M. McNelis, D. Bazow and U. Heinz, arXiv:2101.02827 (2021)

This repository is currently under development, but you can run `test_ode_solver.py` to generate an example plot (you can also edit `solution` in `exact_solution.py`).

Note to self: need to update `evolve.py`, `test_efficiency_implicit.py`

## Runge-Kutta methods

A list of Runge-Kutta methods in Butcher tableau notation can be found in `tables/butcher`. They are organized in two categories: `standard` for RKM and SDRK, and `embedded` for ERK.

Note: we do not consider multiple secondary pairs in the embedded schemes (e.g. Bogacki-Shampine 5(4) does not include the second 4th-order pair).

## Multistep methods

We have also included linear multistep methods such as Adams-Bashforth-Moulton and Numerical Differentiation Formula. Their tables can be found in `tables/multistep`.

Note: currently, the multistep methods only use a fixed time step.

## Parameters

You can adjust the runtime parameters in `parameters.py`. The most important ones are:
    
    adaptive = adaptive time step algorithm (RKM, ERK, SDRK)
    method   = code label of ODE solver     (listed in the READMEs in tables/butcher, tables/multistep)
    
By default, the code uses the Dormand-Prince 5(4) method (adaptive = 'ERK', method = 'DP54'). 

## Status

We have started adding implicit Runge-Kutta methods, linear multistep methods, and interpolators for dense output. 

Note: fully implicit Runge-Kutta is not ready yet.



