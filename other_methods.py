#!/usr/bin/env python3
import numpy as np
from precision import precision     # for myfloat

myfloat = type(precision(1))


# note: these are leftover methods that I left out of the main dictionaries in butcher_tables.py

# 1) Run this script to write these tables to file (you can also include your own)
# 2) Edit the dictionaries in butcher_tables.py to use these tables in the ODE solver


# standard RK methods:

ralston_2 = np.array([
                [0, 0, 0],
                [2/3, 2/3, 0],
                [1, 1/4, 3/4]], dtype = myfloat)

runge_kutta_3 = np.array([
                [0, 0, 0, 0],
                [1/2, 1/2, 0, 0],
                [1, -1, 2, 0],
                [1, 1/6, 2/3, 1/6]], dtype = myfloat)


# ssp = strong stability preserving
ssp_runge_kutta_3 = np.array([
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1/2, 1/4, 1/4, 0],
                [1, 1/6, 1/6, 2/3]], dtype = myfloat)

ssp_runge_kutta_4 = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1/6, 1/6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1/3, 1/6, 1/6, 0, 0, 0, 0, 0, 0, 0, 0],
                [1/2, 1/6, 1/6, 1/6, 0, 0, 0, 0, 0, 0, 0],
                [2/3, 1/6, 1/6, 1/6, 1/6, 0, 0, 0, 0, 0, 0],
                [1/3, 1/15, 1/15, 1/15, 1/15, 1/15, 0, 0, 0, 0, 0],
                [1/2, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 0, 0, 0, 0],
                [2/3, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 1/6, 0, 0, 0],
                [5/6, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 1/6, 1/6, 0, 0],
                [1, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 1/6, 1/6, 1/6, 0],
                [1, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]], dtype = myfloat)

three_eights_rule_4 = np.array([
                [0, 0, 0, 0, 0],
                [1/3, 1/3, 0, 0, 0],
                [2/3, -1/3, 1, 0, 0],
                [1, 1, -1, 1, 0],
                [1, 1/8, 3/8, 3/8, 1/8]], dtype = myfloat)

# from Verner 1978 paper
verner_5 = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [1/18, 1/18, 0, 0, 0, 0, 0],
                [1/6, -1/12, 1/4, 0, 0, 0, 0],
                [2/9, -2/81, 4/27, 8/81, 0, 0, 0],
                [2/3, 40/33, -4/11, -56/11, 54/11, 0, 0],
                [1, -369/73, 72/73, 5380/219, -12285/584, 2695/1752, 0],
                [1, 3/80, 0, 4/25, 243/1120, 77/160, 73/700]], dtype = myfloat)

# Shanks pseudo 8th order (10 stages)
shanks_pseudo_8 = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.14814814814814814, 0.14814814814814814, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.2222222222222222, 0.05555555555555555, 0.16666666666666666, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.3333333333333333, 0.08333333333333333, 0, 0.25, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.125, 0, 0, 0.375, 0, 0, 0, 0, 0, 0],
                [0.6666666666666666, 0.24074074074074073, 0, -0.5, 0.7777777777777778, 0.14814814814814814, 0, 0, 0, 0, 0],
                [0.16666666666666666, 0.09004629629629629, 0, -0.0125, 0.22361111111111112, -0.19074074074074074, 0.05625, 0, 0, 0, 0],
                [1, -11.55, 0, 4.05, -58.2, 32.8, -6.1, 40, 0, 0, 0],
                [0.8333333333333334, -0.4409722222222222, 0, 0.0625, -2.3541666666666665, 1.5833333333333333, -0.03125, 2, 0.013888888888888888, 0, 0],
                [1, 1.8060975609756098, 0, -0.09878048780487805, 8.663414634146342, -4.117073170731707, 0.08780487804878048, -6.146341463414634, -0.07317073170731707, 0.8780487804878049, 0],
                [1, 0.04880952380952381, 0, 0, 0.03214285714285714, 0.3238095238095238, 0.03214285714285714, 0.2571428571428571, 0, 0.2571428571428571,  0.04880952380952381]], dtype = myfloat)


# embedded RK methods:

fehlberg_1_2 = np.array([
                [0, 0, 0, 0],
                [1/2, 1/2, 0, 0],
                [1, 1/256, 255/256, 0],
                [1, 1/256, 255/256, 0],
                [1, 1/512, 255/256, 1/512]], dtype = myfloat)

# from Verner 1978 paper
verner_5_6 = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1/18, 1/18, 0, 0, 0, 0, 0, 0, 0],
                [1/6, -1/12, 1/4, 0, 0, 0, 0, 0, 0],
                [2/9, -2/81, 4/27, 8/81, 0, 0, 0, 0, 0],
                [2/3, 40/33, -4/11, -56/11, 54/11, 0, 0, 0, 0],
                [1, -369/73, 72/73, 5380/219, -12285/584, 2695/1752, 0, 0, 0],
                [8/9, -8716/891, 656/297, 39520/891, -416/11, 52/27, 0, 0, 0],
                [1, 3015/256, -9/4, -4219/78, 5985/128, -539/384, 0, 693/3328, 0],
                [1, 3/80, 0, 4/25, 243/1120, 77/160, 73/700, 0, 0],
                [1, 57/640, 0, -16/65, 1377/2240, 121/320, 0, 891/8320, 2/35]], dtype = myfloat)



standard_other = {'ralston_2':            [ralston_2,            'R2'],
                  'runge_kutta_3':        [runge_kutta_3,        'RK3'],
                  'ssp_runge_kutta_3':    [ssp_runge_kutta_3,    'SSPRK3'],
                  'ssp_runge_kutta_4':    [ssp_runge_kutta_4,    'SSPRK4'],
                  'three_eights_rule_4':  [three_eights_rule_4,  'TER4'],
                  'verner_5':             [verner_5,             'V5'],
                  'shanks_pseudo_8':      [shanks_pseudo_8,      'SP8']}

embedded_other = {'fehlberg_1_2':         [fehlberg_1_2,         'F12'],
                  'verner_5_6':           [verner_5_6,           'V56']}


def debug_table(method, butcher):
    # for testing table properties (within numerical precision):
    #   sum_i b_i = 1
    #   sum_j a_ij = c_i (not a strict condition)

    first_column = butcher[:,0]                                         # get the first column
    reduce_block = np.sum(butcher[:,1:], axis = 1)                      # for each row, sum the remaining columns

    error = np.linalg.norm(reduce_block - first_column, ord = np.inf)   # take the max error element

    if error > 1.e-14:
        print('\ndebug_table warning:', method, 'table does not satisfy usual conditions, error = %.3e (debug table)\n' % error)



# write butcher tables to file:

def main():

        for method in standard_other:

                table = standard_other[method][0]

                print(method)
                debug_table(method, table)

                np.savetxt('butcher_tables/standard/' + method + '.dat', table)


        for method in embedded_other:

                table = embedded_other[method][0]

                print(method)
                debug_table(method, table)

                np.savetxt('butcher_tables/embedded/' + method + '.dat', table)


if __name__ == "__main__":
    main()











