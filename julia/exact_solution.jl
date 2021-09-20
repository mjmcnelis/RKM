#!/usr/bin/env julia

module ExactSolution

include("precision.jl")                         # put inside module to use Precision

export t0, tf, solution, y_exact, y_prime



# adjust exact solution here:
#-------------------------------------
solution = "logistic"                           
#-------------------------------------



A = Precision(100.0)                            # for sine function
cycles = Precision(20.0) 

B = Precision(0.5)                              # for logistic function

t0 = Precision(-10.0)                           # initial and final times
tf = Precision(10.0)

if solution == "sine"

    t0 = Precision(0.0)
    tf = Precision(2.0*pi/A * cycles)

elseif solution == "inverse_power"

    # t0 = Precision(0.0001)
    t0 = Precision(1.0)

elseif solution ∈ ["projectile", "projectile_damped"]

    t0 = Precision(0.0)
    tf = Precision(20.0)
end


function y_exact(t)                             # exact solution (must be column vector format)

    if solution == "logistic"

        return [exp(t) / (1.0 + exp(t)) - B]

    elseif solution == "gaussian"

        return [exp(-t^2)]

    elseif solution == "inverse_power"

        return [t^(-2)]

    elseif solution == "sine"

        return [sin(A * t); A * cos(A * t)]     

    elseif solution == "exponential"
    
        return [exp(10.0 * t)]
    end
end 


function y_prime(t, y)                          # corresponding ODE (needs to agree with above format)

    if solution == "logistic"

        return (y .+ B) .* (1.0 .- y .- B)      # read up on the dot thing
 
    elseif solution == "gaussian"

        return -2.0 * t * y                     # can multiply scalars to vector, so don't need the dot

    elseif solution == "inverse_power"

        return -2.0 * y.^1.5                    # need dot to do this operation on vector

    elseif solution == "sine"

        return [y[2]; -A^2 * y[1]]                # remember indexing in Julia starts with 1, not 0

    elseif solution == "exponential"

        return 10.0 * y
    end
end 


end # module