'''
Modulo Matemático
En honor y apreciación de Jean-Victor Poncelet.  

'''
from numpy import array, concatenate, zeros, linspace, transpose, abs, float64
from numpy.linalg import norm, solve, LinAlgError
import matplotlib.pyplot as plt

def Jacobian(f, x, h, F, U_n, ti, tf):
    n = len(x)
    m = len(f(x, F, U_n, ti, tf))

    J = zeros((m, n))
    delta_x = zeros(n)

    for i in range(0, n):
        delta_x[i] = h
        J[:, i] = (f(x+delta_x, F, U_n, ti, tf) - f(x-delta_x, F, U_n, ti, tf))/(2*h)
        delta_x[i] = 0

    return J

def Newton(f, x, h, N, newton_tol, F, U_n, ti, tf, verbose=False):

    x_n = array(x, copy=True)
    for n in range(0, N):
        J_n = Jacobian(f, x_n, h, F, U_n, ti, tf)
        f_n = f(x_n, F, U_n, ti, tf)
        try:
            delta_x = solve(J_n, -f_n)
        except LinAlgError:
            verbose and print("Fatal error algebraico en NR")
            break
        
        x_n = x_n + delta_x

        if(norm(delta_x)<newton_tol):
            verbose and print(f"Alcanzado newton en {n} iteraciones ") #Printea solo si verbose=True
            break

    return x_n


def Stability_Region(Scheme, N, x0, xf, y0, yf):

    x = linspace(x0, xf, N)
    y = linspace(y0, yf, N)
    rho = zeros((N, N), float64)

    for i in range(N):
        for j in range(N):
            w = complex(x[i], y[j])

            # ecuación test: u' = w u
            F = lambda u, t: w*u

            r = Scheme(1., 1., 0., F)   # dt = 1

            rho[i, j] = abs(r)

    return rho, x, y

def Stability_Region_MC(A,B):
    "Se está cocinando. Dejad cocinar."