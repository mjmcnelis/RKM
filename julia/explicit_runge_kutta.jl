#!/usr/bin/env julia

v(x) = (println(x); x)	

function RK_standard(y, dy1, t, dt, y_prime, butcher; embedded = false)
    
    # y        = current solution y_n
    # dy1      = first intermediate Euler step Δy_n^{(1)}
    # t        = current time t_n
    # dt       = current stepsize dt_n
    # y_prime  = source function f
    # butcher  = Butcher table
    # embedded = if true, return primary/secondary solutions (and last stage for possible FSAL)

    # todo: do embedded later

    if embedded                                            # get c_i, A_ij, b_i and number of stages from Butcher table

        # c = butcher[:-2, 0]
        # A = butcher[:-2, 1:]
        # b = butcher[-2, 1:]
        # bhat = butcher[-1, 1:]
        # stages = butcher.shape[0] - 2
        v("nothing here yet")
        exit()

    else

        c = butcher[1:(end-1),1]
        A = butcher[1:(end-1), 2:end]
        b = butcher[end, 2:end]
        stages = size(butcher)[1] - 1
    end

    dimensions = size(y)[1]
    dy_array = zeros(stages, dimensions)                    # guess make a column vector (or make it stages x dimensions)

    dy_array[1, 1:end] = dy1                                # first stage

    for i in 2:stages                                       # compute remaining stages

        dy = zeros(1, dimensions)

        for j in 1:i-1

            dy .+= A[i,j] * dy_array[j] 
        end

        dy_array[i, 1:end] = dt * y_prime(t + dt*c[i], y + dy)
    end

    dy = zeros(1, dimensions)
    
    for j in 1:stages

        dy .+= b[j] * dy_array[j]
    end

    return y + dy

        # dy_array = [0] * stages
        # dy_array[0] = dy1                                       # first intermediate Euler step
    
        # for i in range(1, stages):                              # loop over remaining intermediate Euler steps
    
        #     dy = 0
    
        #     for j in range(0, i):
    
        #         dy += dy_array[j] * A[i,j]
    
        #     dy_array[i] = dt * y_prime(t + dt*c[i], y + dy)
    
        # dy = 0
    
        # for j in range(0, stages):                              # primary RK iteration (Butcher notation)
    
        #     dy += dy_array[j] * b[j]
    
        # if embedded:                                            # secondary RK iteration (for embedded RK)
    
        #     dyhat = 0
    
        #     for j in range(0, stages):
    
        #         dyhat += dy_array[j] * bhat[j]
    
        #     k_last = dy_array[-1] / dt
    
        #     return (y + dyhat), (y + dy), k_last                # updated ERK solutions (secondary, primary, last stage)
    
        # return y + dy                                           # updated solution (primary)
end