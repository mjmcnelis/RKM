#!/usr/bin/env julia

module ExactSolution

    include("precision.jl")                         # put inside module to use Precision

    export t0, tf, solution, y_exact, y_prime

    solution = "logistic"                        # adjust exact solution

    A = Precision(100.0)                            # for sine function
    cycles = Precision(20.0) 

    B = Precision(0.5)                              # for logistic function

    t0 = Precision(-10.0)                           # initial and final times
    tf = Precision(10.0)

    if solution == "sine"

        t0 = Precision(0.0)
        tf = Precision(2.0*pi/A * cycles)

    elseif solution == "inverse_power"

        t0 = Precision(0.0001)

    elseif solution ∈ ["projectile", "projectile_damped"]

        t0 = Precision(0.0)
        tf = Precision(20.0)
    end
    

    function y_exact(t)                             # exact solution

        if solution == "logistic"

            return exp(t) / (1.0 + exp(t)) - B

        elseif solution == "gaussian"

            return exp(-t^2)

        elseif solution == "inverse_power"

            return t^(-2)

        elseif solution == "sine"

            return [sin(A*t); A * cos(A*t)]         # want this to be a column vector?

        elseif solution == "exponential"
        
            return exp(10.0 * t)
        end
    end 


    function y_prime(t, y)                          # corresponding ODE

        if solution == "logistic"

            return (y + B) * (1.0 - y - B)

        elseif solution == "gaussian"

            return -2.0 * t * y

        elseif solution == "inverse_power"

            return -2.0 * y^1.5

        elseif solution == "sine"

            return [y[2]; -A * y[1]]                # remember index starts with 1, not 0

        elseif solution == "exponential"

            return 10.0 * y
        end
    end 


end # module