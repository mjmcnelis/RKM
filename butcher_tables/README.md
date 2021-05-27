These are all of the Runge-Kutta methods I have compiled so far:

## Low order (1-3)
<table>
<tr valign="top"><td>

|      Standard       |        |
|:-------------------:|:------:|
| Euler 1             | E1     |
| Heun 2              | H2     |
| Midpoint 2          | M2     |
| Ralston 2           | R2     |
| Heun 3              | H3     |
| Ralston 3           | R3     |
| Runge Kutta 3       | RK3    |
| SSP Runge Kutta 3   | SSPRK3 |

</td><td valign="top">

|      Embedded         |      |
|:---------------------:|:----:|
| Fehlberg 1(2)         | F12  |
| Heun Euler 2(1)       | HE21 |
| Bogacki Shampine 3(2) | BS32 |

</td></tr> </table>

Note: SSP = strong stability preserving

## Medium order (4-6)
<table>
<tr valign="top"><td>

|      Standard       |        |
|:-------------------:|:------:|
| Runge Kutta 4       | RK4    |
| Three Eights Rule 4 | TER4   |
| Ralston 4           | R4     |
| SSP Runge Kutta 4   | SSPRK4 |
| Fehlberg 4          | F4     |
| Butcher 5           | B5     |
| Cash Karp 5         | CK5    |
| Dormand Prince 5    | DP5    |
| Bogacki Shampone 5  | BS5    |
| Tsitouras 5         | T5     |
| Verner 5            | V5     |
| Butcher 6           | B6     |
| Verner 6            | V6     |
  
</td><td valign="top">

|      Embedded         |      |
|:---------------------:|:----:|
| Fehlberg 4(5)         | F45  |
| Cash Karp 5(4)        | CK54 |
| Dormand Prince 5(4)   | DP54 |
| Bogacki Shampine 5(4) | BS54 |
| Tsitouras 5(4)        | T54  |
| Verner 5(6)           | V56  |
| Verner 6(5)           | V65  |

</td></tr> </table>

Note: the embedded schemes in `standard` have their secondary method removed, along with extraneous stages (e.g. F4 uses the fourth-order method and has 5 stages instead of 6). 

## High order (7-9)
<table>
<tr valign="top"><td>

|      Standard       |      |
|:-------------------:|:----:|
| Fehlberg 7          | F7   |
| Curtis 8            | C8   |
| Shanks 8            | S8   |
| Shanks Pseudo 8     | SP8  |
| Dormand Prince 8    | DP8  |

</td><td valign="top">

|      Embedded         |      |
|:---------------------:|:----:|
| Fehlberg 7(8)         | F78  |
| Dormand Prince 8(7)   | DP87 |

</td></tr> </table>


## Very high order (10+)
<table>
<tr valign="top"><td>

|      Standard       |      |
|:-------------------:|:----:|
| Feagin 10           | F10  |
| Feagin 14           | F14  |

</td><td valign="top">

|      Embedded         |      |
|:---------------------:|:----:|
| Feagin 10(8)          | F108 |
| Feagin 14(12)         | F1412|

</td></tr> </table>

Note: Feagin's 10th and 14th order tables are not yet ready for high precision runs.
