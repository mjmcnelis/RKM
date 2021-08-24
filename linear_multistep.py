#!/usr/bin/env python3
import math
import numpy as np
from implicit_runge_kutta import EPS_ROOT, ITERATIONS

# these are coefficients for ABM extrapolation (see pg.9 on https://www.asc.tuwien.ac.at/~melenk/teach/num_DGL_SS16/MatlabODESuite.pdf)
ABM_extrapolation_coefficients = [-1/2, -1/6, -1/10, -19/270, -27/502, -863/19950]    



def adams_bashforth(y, t, dt, f_list, y_prime, adams, steps):

    # y = current variable y_n
    # t = current time t_n
    # f_list = [y'_{n-1}, y'_{n-2}, ...]                

    f = y_prime(t, y)

    dy = adams[0] * dt * f

    for i in range(0, steps):
        dy += adams[i+1] * dt * f_list[i]

    evaluations = 1

    return y + dy, f, evaluations



def adams_moulton(y, t, dt, f_list, y_prime, jacobian, adams, steps, root, 
                  eps_root = EPS_ROOT, max_iterations = ITERATIONS, norm = None):

    f = 0
    z0 = 0
    evaluations = 0
    dimension = y.shape[0]
    identity = np.identity(dimension)	


    # check again if f_list is correct or not

    if steps > 0:                                           # compute z0, which is the fixed part of z = dy
        f = y_prime(t, y) 
        z0 += adams[1] * dt * f
        evaluations += 1

    for i in range(1, steps):
        z0 += adams[i+1] * dt * f_list[i-1] 

    z = 0                                                   # z = dy
    z_prev = 0

    if root is 'newton_fast':
        J = identity  -  adams[0] * dt * jacobian(t, y)		# only evaluate jacobian once
        evaluations += dimension

    for n in range(0, max_iterations):                         

        if root is not 'fixed_point':                       # solve nonlinear system g(z) = 0 via newton
            g = z  -  z0  -  adams[0] * dt * y_prime(t + dt, y + z)	

            evaluations += 1

            if root is 'newton':							# evaluate jacobian for every iteration
                J = identity  -  adams[0] * dt * jacobian(t + dt, y + z)

                evaluations += dimension
            


            # todo: compute inverse jacobian if newton_fast	
            dz = np.linalg.solve(J.astype('float64'), -g.astype('float64'))	



            z += dz											# newton iteration dz = -J^{-1}.g (linalg only supports float64)

            delta = np.linalg.norm(g, ord = norm)
            tolerance = eps_root

            if delta <= tolerance:							# check for convergence of solution g(z) = 0
                break

        else:                                               # fixed-point iteration 
            z = z0  +  adams[0] * dt * y_prime(t + dt, y + z)

            evaluations += 1
        
            delta   = np.linalg.norm(z - z_prev, ord = norm)	
            dy_norm = np.linalg.norm(z, ord = norm)
        
            tolerance = eps_root * dy_norm

            if delta <= tolerance:						    # check for convergence of solution
                break
            
            z_prev = z.copy()

    dy = z

    return y + dy, f, evaluations



def adams_bashforth_moulton(y, t, dt, f_list, y_prime, jacobian, adams, steps, root = 'newton_fast', 
                            extrapolate = True, eps_root = EPS_ROOT, max_iterations = ITERATIONS, norm = None):

    # predictor-corrector scheme (with option of extrapolation)
    
    f = y_prime(t, y)
    evaluations = 1
    dimension = y.shape[0]
    identity = np.identity(dimension)	
    
    zP = adams[0,0] * dt * f                                # compute predictor zP = dyP w/ adams-bashforth row adams[0,:]

    for i in range(0, steps):
        zP += adams[0,i+1] * dt * f_list[i]

    #-------------------------------------------------------  

    z0 = 0

    if steps > 0:                                           # compute z0, which is fixed part of z = dy
        z0 += adams[1,1] * dt * f

    for i in range(1, steps):
        z0 += adams[1,i+1] * dt * f_list[i-1] 

    z = zP.copy()                                           # initialize z = dy as predictor value
    z_prev = 0

    if root is 'newton_fast':
        J = identity  -  adams[1,0] * dt * jacobian(t, y)	# only evaluate jacobian once (should t-argument be t or t+dt?)
        evaluations += dimension

    for n in range(0, max_iterations):                      # compute corrector w/ adams-moulton row adams[1,:]
     
        if root is not 'fixed_point':                       # solve nonlinear system g(z) = 0 via newton
            g = z  -  z0  -  adams[1,0] * dt * y_prime(t + dt, y + z)	

            evaluations += 1

            if root is 'newton':							# evaluate jacobian for every iteration
                J = identity  -  adams[1,0] * dt * jacobian(t + dt, y + z)

                evaluations += dimension
            


            # todo: compute inverse jacobian if newton_fast	
            dz = np.linalg.solve(J.astype('float64'), -g.astype('float64')) 



            z += dz											# newton iteration dz = -J^{-1}.g (linalg only supports float64)

            delta = np.linalg.norm(g, ord = norm)
            tolerance = eps_root

            if delta <= tolerance:							# check for convergence of solution g(z) = 0
                break
                
        else:                                               # fixed-point iteration 
            z = z0  +  adams[1,0] * dt * y_prime(t + dt, y + z)   

            evaluations += 1

            delta   = np.linalg.norm(z - z_prev, ord = norm)	
            dy_norm = np.linalg.norm(z, ord = norm)
        
            tolerance = eps_root * dy_norm

            if delta <= tolerance:						    # check for convergence of solution
                break

            z_prev = z.copy()          

    if extrapolate:
        coefficient = ABM_extrapolation_coefficients[steps] # order = steps + 1
        z += coefficient * (z - zP)                         # extrapolate ABM corrector

    dy = z

    return y + dy, f, evaluations



def compute_DF_predictor(y, y_list, df):

    coeff = df[1,:]                                         # get coefficients of differentiation formula predictor (second row of DF table)

    dy_0 = coeff[0] * y

    for i in range(0, len(coeff) - 1):
        dy_0 += coeff[i+1] * y_list[i]

    return dy_0



def differentiation_formula(y, t, dt, y_list, y_prime, jacobian, df, steps, root, 
                            dy_0 = 0, eps_root = EPS_ROOT, max_iterations = ITERATIONS, norm = None):

    # y_list = [y_{n-1}, y_{n-2}, ...]
    # df = table of differentiation formula (either BDF or NDF)

    evaluations = 0
    dimension = y.shape[0]                                  # subtracted 1 from DF coefficient of y = y_n to get  
    identity = np.identity(dimension)                       # dy = y_{n+1} - y = df[0,0].dt.f(t+dt,y+dy) + (df[0,1]-1).y + df[0,2].y_{n-1} + ...

    z0 = (df[0,1] - 1) * y                                  # compute z0, which is the fixed part of z = dy 
 
    for i in range(0, steps):
        z0 += df[0, i+2] * y_list[i] 

    z = 0                                                   # z = dy
    z_prev = 0

    z = dy_0

    if root is 'newton_fast':
        J = identity  -  df[0,0] * dt * jacobian(t, y)		# only evaluate jacobian once
        evaluations += dimension

    for n in range(0, max_iterations):                         

        if root is not 'fixed_point':                       # solve nonlinear system g(z) = 0 via newton
            g = z  -  z0  -  df[0,0] * dt * y_prime(t + dt, y + z)	
            evaluations += 1

            if root is 'newton':							# evaluate jacobian for every iteration
                J = identity  -  df[0,0] * dt * jacobian(t + dt, y + z)
                evaluations += dimension
            
            # todo: compute inverse jacobian if newton_fast	
            dz = np.linalg.solve(J.astype('float64'), -g.astype('float64'))	

            z += dz											# newton iteration dz = -J^{-1}.g (linalg only supports float64)

            delta = np.linalg.norm(g, ord = norm)
            tolerance = eps_root

            if delta <= tolerance:							# check for convergence of solution g(z) = 0
                break
            
        else:                                               # fixed-point iteration 
            z = z0  +  df[0,0] * dt * y_prime(t + dt, y + z)

            evaluations += 1
        
            delta   = np.linalg.norm(z - z_prev, ord = norm)	
            dy_norm = np.linalg.norm(z, ord = norm)
        
            tolerance = eps_root * dy_norm

            if delta <= tolerance:						    # check for convergence of solution
                break
            
            z_prev = z.copy()

    dy = z

    return y + dy, y, evaluations
