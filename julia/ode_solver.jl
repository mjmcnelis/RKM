#!/usr/bin/env julia

include("precision.jl")

v(x) = (println(x); x)	


function ode_solver(y0, t0::Precision, tf::Precision, n_max = 10000)

    y = copy(y0)
    t = t0

    dt0::Precision = 0.001

    dt = dt0

    v(typeof(dt0))

    finish = false

    for n in 1:n_max

        y = (1. + dt)y
        
        if t >= tf
            finish = true
            break
        end

        t += dt

    end

    if finish
        println("\node_solver took X seconds to finish\n")
    else   
        println("\node_solver flag: evolution stopped at t = Y seconds\n")
    end

    return y, t

end






