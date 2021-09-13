#!/usr/bin/env python3
import numpy as np
from precision import myfloat


# Adams-Bashforth (orders 2-5)
#--------------------------------------------------------------------------------------------------
adams_bashforth_1 = np.array([
        [1]], dtype = myfloat)

adams_bashforth_2 = np.array([
        [3/2, -1/2]], dtype = myfloat)

adams_bashforth_3 = np.array([
        [23/12, -16/12, 5/12]], dtype = myfloat)

adams_bashforth_4 = np.array([
        [55/24, -59/24, 37/24, -9/24]], dtype = myfloat)

adams_bashforth_5 = np.array([
        [1901/720, -2774/720, 2616/720, -1274/720, 251/720]], dtype = myfloat)

adams_bashforth_6 = np.array([
        [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440]], dtype = myfloat)

adams_bashforth_8 =  np.array([
        [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960]], dtype = myfloat)
#--------------------------------------------------------------------------------------------------



# Adams-Moulton (orders 2-5)
#--------------------------------------------------------------------------------------------------
adams_moulton_1 = np.array([
        [1]], dtype = myfloat)

adams_moulton_2 = np.array([
        [1/2, 1/2]], dtype = myfloat)

adams_moulton_3 = np.array([
        [5/12, 8/12, -1/12]], dtype = myfloat)

adams_moulton_4 = np.array([
        [9/24, 19/24, -5/24, 1/24]], dtype = myfloat)

adams_moulton_5 = np.array([
        [251/720, 646/720, -264/720, 106/720, -19/720]], dtype = myfloat)

adams_moulton_6 = np.array([
        [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]], dtype = myfloat)

adams_moulton_8 =  np.array([
        [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960]], dtype = myfloat)
#--------------------------------------------------------------------------------------------------



# Adams-Bashforth-Moulton predictor-corrector (orders 2-5)
#--------------------------------------------------------------------------------------------------
# first row  = coefficients of AB predictor
# second row = coefficients of AM corrector

adams_bashforth_moulton_1 = np.array([
        [1],
        [1]], dtype = myfloat)

adams_bashforth_moulton_2 = np.array([
        [3/2, -1/2],
        [1/2, 1/2]], dtype = myfloat)

adams_bashforth_moulton_3 = np.array([
        [23/12, -16/12, 5/12],
        [5/12, 8/12, -1/12]], dtype = myfloat)

adams_bashforth_moulton_4 = np.array([
        [55/24, -59/24, 37/24, -9/24],
        [9/24, 19/24, -5/24, 1/24]], dtype = myfloat)

adams_bashforth_moulton_5 = np.array([
        [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
        [251/720, 646/720, -264/720, 106/720, -19/720]], dtype = myfloat)

adams_bashforth_moulton_6 = np.array([
        [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440],
        [475/1440, 1427/1440, -798/1440, 482/1440, -173/1440, 27/1440]], dtype = myfloat)

adams_bashforth_moulton_8 =  np.array([
        [434241/120960, -1152169/120960, 2183877/120960, -2664477/120960, 2102243/120960, -1041723/120960, 295767/120960, -36799/120960],
        [36799/120960, 139849/120960, -121797/120960, 123133/120960, -88547/120960, 41499/120960, -11351/120960, 1375/120960]], dtype = myfloat)
#--------------------------------------------------------------------------------------------------



# Backward Differentiation Formula (orders 2-6)
#--------------------------------------------------------------------------------------------------
# first row  = coefficients of BDF corrector
# second row = coefficients of BDF predictor

backward_differentiation_formula_1 = np.array([
        [1, 1],
        [1, -1]], dtype = myfloat)

backward_differentiation_formula_2 = np.array([
        [2/3, 4/3, -1/3],
        [2, -3, 1]], dtype = myfloat)

backward_differentiation_formula_3 = np.array([
        [6/11, 18/11, -9/11, 2/11],
        [3, -6, 4, -1]], dtype = myfloat)

backward_differentiation_formula_4 = np.array([
        [12/25, 48/25, -36/25, 16/25, -3/25],
        [4, -10, 10, -5, 1]], dtype = myfloat)

backward_differentiation_formula_5 = np.array([
        [60/137, 300/137, -300/137, 200/137, -75/137, 12/137],
        [5, -15, 20, -15, 6, -1]], dtype = myfloat)

backward_differentiation_formula_6 = np.array([
        [60/147, 360/147, -450/147, 400/147, -225/147, 72/147, -10/147],
        [6, -21, 35, -35, 21, -7, 1]], dtype = myfloat)
#--------------------------------------------------------------------------------------------------



# Numerical Differentiation Formula (should I use same predictor as BDF?)
#--------------------------------------------------------------------------------------------------
# kappa_k = [-37/200, -1/9, -0.0823, -0.0415]
# gamma_k = sum^k_{n=1} 1/n = [1, 3/2, 11/6, 25/12]
kg1 = -37/200
kg2 = -1/9 * 3/2
kg3 = -0.0823 * 11/6
kg4 = -0.0415 * 25/12

# note: ode solver does not access zero element in second row

numerical_differentiation_formula_1 = np.array([
        [1/(1 - kg1), (1 - 2*kg1)/(1 - kg1), kg1/(1 - kg1)],
        [0, 1, -1]], dtype = myfloat)

numerical_differentiation_formula_2 = np.array([
        [(2/3)/(1 - kg2), (4/3 - 3*kg2)/(1 - kg2), -(1/3 - 3*kg2)/(1 - kg2), -kg2/(1 - kg2)],
        [0, 2, -3, 1]], dtype = myfloat)

numerical_differentiation_formula_3 = np.array([
        [(6/11)/(1 - kg3), (18/11 - 4*kg3)/(1 - kg3), -(9/11 - 6*kg3)/(1 - kg3), (2/11 - 4*kg3)/(1 - kg3), kg3/(1 - kg3)],
        [0, 3, -6, 4, -1]], dtype = myfloat)

numerical_differentiation_formula_4 = np.array([
        [12/25/(1 - kg4), (48/25 - 5*kg4)/(1 - kg4), -(36/25 - 10*kg4)/(1 - kg4), (16/25 - 10*kg4)/(1 - kg4), -(3/25 - 5*kg4)/(1 - kg4), -kg4/(1 - kg4)],
        [0, 4, -10, 10, -5, 1]], dtype = myfloat)


# dictionaries
#--------------------------------------------------------------------------------------------------
adams_explicit_dict = {'AB1':  ['adams_bashforth_1', adams_bashforth_1],        # explicit euler
                       'AB2':  ['adams_bashforth_2', adams_bashforth_2],
                       'AB3':  ['adams_bashforth_3', adams_bashforth_3],
                       'AB4':  ['adams_bashforth_4', adams_bashforth_4],
                       'AB5':  ['adams_bashforth_5', adams_bashforth_5],
                       'AB6':  ['adams_bashforth_6', adams_bashforth_6],
                       'AB8':  ['adams_bashforth_8', adams_bashforth_8]}        # todo: having trouble finding other tables

adams_implicit_dict = {'AM1':  ['adams_moulton_1', adams_moulton_1],            # implicit euler
                       'AM2':  ['adams_moulton_2', adams_moulton_2],            # implicit trapezoid
                       'AM3':  ['adams_moulton_3', adams_moulton_3],
                       'AM4':  ['adams_moulton_4', adams_moulton_4],
                       'AM5':  ['adams_moulton_5', adams_moulton_5],
                       'AM6':  ['adams_moulton_6', adams_moulton_6],
                       'AM8':  ['adams_moulton_8', adams_moulton_8]}

adams_predictor_corrector_dict = {
                       'ABM1': ['adams_bashforth_moulton_1', adams_bashforth_moulton_1],
                       'ABM2': ['adams_bashforth_moulton_2', adams_bashforth_moulton_2],
                       'ABM3': ['adams_bashforth_moulton_3', adams_bashforth_moulton_3],
                       'ABM4': ['adams_bashforth_moulton_4', adams_bashforth_moulton_4],
                       'ABM5': ['adams_bashforth_moulton_5', adams_bashforth_moulton_5],
                       'ABM6': ['adams_bashforth_moulton_6', adams_bashforth_moulton_6],
                       'ABM8': ['adams_bashforth_moulton_8', adams_bashforth_moulton_8]}

backward_differentiation_formula_dict = {
                       'BDF1': ['backward_differentiation_formula_1', backward_differentiation_formula_1],     # implicit euler
                       'BDF2': ['backward_differentiation_formula_2', backward_differentiation_formula_2],
                       'BDF3': ['backward_differentiation_formula_3', backward_differentiation_formula_3],
                       'BDF4': ['backward_differentiation_formula_4', backward_differentiation_formula_4],
                       'BDF5': ['backward_differentiation_formula_5', backward_differentiation_formula_5],
                       'BDF6': ['backward_differentiation_formula_6', backward_differentiation_formula_6]}

numerical_differentiation_formula_dict = {
                       'NDF1': ['numerical_differentiation_formula_1', numerical_differentiation_formula_1],
                       'NDF2': ['numerical_differentiation_formula_2', numerical_differentiation_formula_2],
                       'NDF3': ['numerical_differentiation_formula_3', numerical_differentiation_formula_3],
                       'NDF4': ['numerical_differentiation_formula_4', numerical_differentiation_formula_4]}


def debug_table(method, table):

    # todo: need to analyze error by row to account for ABM tables (something like in butcher)

    error = abs(np.sum(table)) - 1

    if error > 1.e-14:
        print('\ndebug_table warning:', method, 'table does not satisfy usual conditions, error = %.3e (may need to debug table)\n' % error)



def main():
        for label in adams_explicit_dict:                        # write tables to file

                method = adams_explicit_dict[label][0]
                table  = adams_explicit_dict[label][1]

                np.savetxt('tables/multistep/adams/adams_bashforth/' + method + '.dat', table)

                debug_table(method, table)

                print(method)

        for label in adams_implicit_dict:

                method = adams_implicit_dict[label][0]
                table  = adams_implicit_dict[label][1]

                np.savetxt('tables/multistep/adams/adams_moulton/' + method + '.dat', table)

                debug_table(method, table)

                print(method)

        for label in adams_predictor_corrector_dict:

                method = adams_predictor_corrector_dict[label][0]
                table  = adams_predictor_corrector_dict[label][1]

                np.savetxt('tables/multistep/adams/adams_bashforth_moulton/' + method + '.dat', table)

                # worry about later
                debug_table(method, table)

                print(method)

        for label in backward_differentiation_formula_dict:

                method = backward_differentiation_formula_dict[label][0]
                table  = backward_differentiation_formula_dict[label][1]

                np.savetxt('tables/multistep/backward_differentiation_formula/' + method + '.dat', table)

                debug_table(method, table[:,1:])        # exclude first element for debug

                print(method)

        for label in numerical_differentiation_formula_dict:

                method = numerical_differentiation_formula_dict[label][0]
                table  = numerical_differentiation_formula_dict[label][1]

                np.savetxt('tables/multistep/numerical_differentiation_formula/' + method + '.dat', table)

                debug_table(method, table[:,1:])        # exclude first element for debug

                print(method)



if __name__ == "__main__":
    main()



