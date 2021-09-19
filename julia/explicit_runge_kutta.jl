#!/usr/bin/env julia

v(x) = (println(x); x)	

function RK_standard(y, dy1, t, dt, y_prime; butcher = 0)
    

    return y + dy1
end