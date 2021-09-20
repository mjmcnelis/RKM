#!/usr/bin/env julia
include("ode_solver.jl")
include("parameters.jl")
# using ExactSolution                   # how to do something like this?
include("exact_solution.jl")
using .ExactSolution 

v(x) = (println(x); x)	

y0 = y_exact(t0)

jacobian = 4

y, t, dt, evals, rejection = ode_solver(y0, t0, tf, y_prime, parameters, jacobian = jacobian)

v(y[end])
v(t[end])

# using Plots: plot, savefig
# fig = plot(t[:], y[1,:])                # basic figure 
# # @show fig                             # this didn't plt show after terminal
# savefig(fig, "plot.png")

