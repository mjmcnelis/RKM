
def SDRK_step(y, t, dt, p, l, eps = 1.e-8, low = 0.2, high = 5, S = 0.9):

    rescale = 1

    for i in range(0, 100):

        dt *= rescale

        y1 = RK(y, t, dt)

        y_mid = RK(y, t, dt/2)
        y2 = RK(y_mid, t+dt/2, dt/2)

        error = (y2 - y1) / (2**p - 1)
        yR = y2 + error

        error_norm = np.linalg.norm(error, ord = l)
        y_norm = np.linalg.norm(yR, ord = l)
        dy_norm = np.linalg.norm(yR - y, ord = l)

        tolerance = eps * max(y_norm, dy_norm)

        if error_norm == 0:
            rescale = 1
        else:
            rescale = (tolerance / error_norm)**(1/(1+p))
            rescale = min(high, max(low, S*rescale))

        if error_norm <= tolerance:
            dt_next = dt*rescale

            return yR, dt, dt_next
        else:
            rescale = min(S, rescale)

    dt_next = dt*rescale

    return yR, dt, dt_next




