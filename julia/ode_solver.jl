#!/usr/bin/env julia
include("precision.jl")
include("explicit_runge_kutta.jl")

v(x) = (println(x); x)	

function ode_solver(y0, t0::Precision, tf::Precision, y_prime, parameters; jacobian = 0)
            
    start = time()

    adaptive            = parameters["adaptive"]            # get parameters
    method_label        = parameters["method"]
    dt0::Precision      = parameters["dt_initial"]
    n_max               = parameters["max_steps"]
    eps::Precision      = parameters["eps"]
    norm                = parameters["norm"]
    dt_min::Precision   = parameters["dt_min"]
    dt_max::Precision   = parameters["dt_max"]
    root                = parameters["root"]
    max_iterations      = parameters["max_iterations"]
    eps_root::Precision = parameters["eps_root"]
    free_embedded       = parameters["free_embedded"]

    y = copy(y0)                                            # set initial conditions
    t = t0
    dt = dt0

    y_prev = copy(y0)                                       # for RKM
    dt_next = dt                                            # for ERK/SDRK

    stages = 1  # temp
  
    steps = 0
    evaluations = 0
    total_attempts = 0
    finish = false
  
    y_array = Vector{Precision}()                           # okay for y?
    t_array = Vector{Precision}()
    dt_array = Vector{Precision}()

    for n in 1:n_max

        push!(y_array, y)                                   # append y,t arrays
        push!(t_array, t)

        if adaptive == false                                # use a fixed time step 
        
            dy1 = dt * y_prime(t, y)
            y = RK_standard(y, dy1, t, dt, y_prime)
            
            evaluations += stages
            total_attempts += 1
        
        elseif adaptive == "RKM"

            v("nothing here yet")
            exit()

        elseif adaptive == "ERK"

            v("nothing here yet")
            exit()

        elseif adaptive == "SDRK"
        
            v("nothing here yet")
            exit()
        
        else

            v("error")
            exit()
        end
        
        push!(dt_array, dt)                                 # append dt array

        if t >= tf

            finish = true
            break
        end


        t += dt
        steps += 1

    end

    rejection_rate = 100. * (1  -  (steps + 1)/total_attempts)          

    if finish

        println("\node_solver took $(round(time() - start, sigdigits = 2)) seconds to finish\n")

    else   

        println("\node_solver flag: evolution stopped at t = $(round(t, sigdigits = 3)) seconds\n")
    end

    return y_array, t_array, dt_array, evaluations, rejection_rate
    # return y, t_array, dt_array, evaluations, rejection_rate

end






