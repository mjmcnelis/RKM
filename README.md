**RKM (c) Mike McNelis**

Created on 4/20/2021 by Mike McNelis\
Last edited on 5/18/2021 by Mike McNelis

## Summary
An explicit Runge-Kutta ODE solver with a new adaptive step size method.

The Python code evolves ordinary differential equations with one of three adaptive methods:

| ||
|:----:|:-------------:|
| RKM  | new method    |
| ERK  | embedded      |
| SDRK | step doubling |

The RKM algorithm was first used in one of my papers to solve fluid dynamic equations (see Sec. 3.6 for the derivation)

    M. McNelis, D. Bazow and U. Heinz, arXiv:2101.02827 (2021)
    
This repository is currently under development, but you can run `test.py` to generate an example plot. 


## Runge-Kutta methods

A list of Runge-Kutta methods in Butcher tableau notation can be found in `butcher_tables`. They are organized in two categories: `standard` for RKM and SDRK, and `embedded` for ERK.
   
<table>
<tr valign="top"><td>

|      Standard       |      |
|:-------------------:|:----:|
| Euler 1             | E1   |
| Heun 2              | H2   |
| Midpoint 2          | M2   |
| Ralston 2           | R2   |
| Heun 3              | H3   |
| Ralston 3           | R3   |
| Runge Kutta 3       | RK3  |
| Ralston 4           | R4   |
| Runge Kutta 4       | RK4  |
| 3/8 Rule 4          | TER4 |
| Fehlberg 4          | F4   |
| Butcher 5           | B5   |
| Cash Karp 5         | CK5  |
| Dormand Prince 5    | DP5  |
| Butcher 6           | B6   |
| Verner 6            | V6   |
| Fehlberg 7          | F7   |
| Shanks 8            | S8   |
| Dormand Prince 8    | DP8  |
| Feagin 10           | F10  |
| Feagin 14           | F14  |

</td><td valign="top">

|      Embedded         |      |
|:---------------------:|:----:|
| Fehlberg 1(2)         | F12  |
| Heun Euler 2(1)       | HE21 |
| Bogacki Shampine 3(2) | BS32 |
| Fehlberg 4(5)         | F45  |
| Cash Karp 5(4)        | CK54 |
| Dormand Prince 5(4)   | DP54 |
| Verner 6(5)           | V65  |
| Fehlberg 7(8)         | F78  |
| Dormand Prince 8(7)   | DP87 |
| Feagin 10(8)          | F108 |
| Feagin 14(12)         | F1412|

</td></tr> </table>

Note: Feagin's 10th and 14th order tables are not yet ready for high precision runs. 
    
    
    
