#!/usr/bin/env julia

using Profile

include("ode_solver.jl")

y0 = 1.0
t0 = Precision(0)
tf = Precision(1.0)

@time begin
y, t = ode_solver(y0, t0, tf)
end


println(y, "\t", exp(t), "\t", t)




