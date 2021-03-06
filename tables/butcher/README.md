These are all of the Runge-Kutta methods I have compiled so far:

## Low order (1-3)
<table>
<tr valign="top"><td>

  |  Standard Explicit  |        |
  |:-------------------:|:------:|
  | Euler 1             | E1     |
  | Heun 2              | H2     |
  | Midpoint 2          | M2     |
  | Ralston 2           | R2     |
  | Heun 3              | H3     |
  | Ralston 3           | R3     |
  | Runge Kutta 3       | RK3    |
  | SSP Shu Osher 3     | SSPSO3 |
  | SSP Spiteri Ruuth 3 | SSPSR3 |
  
</td><td valign="top">
  
  |   Embedded Explicit   |      |
  |:---------------------:|:----:|
  | Fehlberg 1(2)         | F12  |
  | Heun Euler 2(1)       | HE21 |
  | Bogacki Shampine 3(2) | BS32 |

</td></tr> </table>

<table>
<tr valign="top"><td>
  
  |   Standard Implicit     |        |
  |:-----------------------:|:------:|
  | Backward Euler 1        | BE1    |
  | Implicit Midpoint 2     | IM2    |
  | Crank Nicolson 2        | CN2    |
  | Qin Zhang 2             | QZ2    |
  | Pareschi Russo 2        | PR2    |
  | Lobatto IIIB 2          | LIIIB2 |
  | Lobatto IIIC 2          | LIIIC2 |
  | Kraaijevanger Spijker 2 | KS2    |
  | Pareschi Russo 3        | PR3    |
  | Crouzeix 3              | C3     |
  | Radau IA 3              | RIA3   |
  | Radau IIA 3             | RIIA3  |
  | DIRK L-Stable 3         | DIRKL3 |
  
</td><td valign="top">

  |   Embedded Implicit   |         |
  |:---------------------:|:-------:|
  | Crank Nicolson 2(1)   | CN21    |
  | Lobatto IIIB 2(1)     | LIIIB21 |
  | Lobatto IIIC 2(1)     | LIIIC21 |
  
</td></tr> </table>

Note: SSP = strong stability preserving

## Medium order (4-6)
<table>
<tr valign="top"><td>

| Standard Explicit   |        |
|:-------------------:|:------:|
| Runge Kutta 4       | RK4    |
| Three Eights Rule 4 | TER4   |
| Ralston 4           | R4     |
| SSP Ketcheson 4     | SSPK4  |
| Fehlberg 4          | F4     |
| Butcher 5           | B5     |
| Cash Karp 5         | CK5    |
| Dormand Prince 5    | DP5    |
| Bogacki Shampine 5  | BS5    |
| Tsitouras 5         | T5     |
| Verner 5            | V5     |
| Butcher 6           | B6     |
| Verner 6            | V6     |
  
</td><td valign="top">
 
|    Embedded Explicit  |      |
|:---------------------:|:----:|
| Fehlberg 4(5)         | F45  |
| Cash Karp 5(4)        | CK54 |
| Dormand Prince 5(4)   | DP54 |
| Bogacki Shampine 5(4) | BS54 |
| Tsitouras 5(4)        | T54  |
| Verner 5(6)           | V56  |
| Verner 6(5)           | V65  |
  
</td></tr> </table>

<table>
<tr valign="top"><td>

| Standard Implicit     |         |
|:---------------------:|:-------:|
| Norsett 4             | N4      |
| Gauss Legendre 4      | GL4     |
| Lobatto IIIA 4        | LIIIA4  |
| Lobatto IIIB 4        | LIIIB4  | 
| Lobatto IIIC 4        | LIIIC4  | 
| Lobatto IIICS 4       | LIIICS4 | 
| Lobatto IIID 4        | LIIID4  | 
| Radau IA 5            | RIA5    |
| Radau IIA 5           | RIIA5   |
| Gauss Legendre 6      | GL6     |

</td><td valign="top">
  
| Embedded Implicit     |          |
|:---------------------:|:--------:|
| Gauss Legendre 4(2)   | GL42     |
| Lobatto IIIA 4(2)     | LIIIA42  |
| Lobatto IIIB 4(2)     | LIIIB42  | 
| Lobatto IIIC 4(2)     | LIIIC42  | 
| Lobatto IIICS 4(2)    | LIIICS42 |
| Radau IIA 5(2)        | RIIA52   |
| Gauss Legendre 6(4)   | GL64     |
  
</td></tr> </table>

Note: the embedded schemes in `standard` have their secondary method removed, along with extraneous stages. 

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
