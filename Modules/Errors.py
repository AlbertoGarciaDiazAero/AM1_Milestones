'''
Docstring for Modules.Errors
'''

from numpy import  zeros, linspace, log, polyfit, log10
from numpy.linalg import norm
from Modules.temporal_schemes import Cauchy_problem


def refine_mesh(t1):
    """
    Dada una malla t1 = [t0, t1, ..., tN], genera 
    t2 = [t0, (t0+t1)/2, t1, (t1+t2)/2, ..., tN].
    """
    N = len(t1) - 1
    t2 = zeros(2*N + 1, dtype=float)

    for i in range(N):
        t2[2*i]   = t1[i]
        t2[2*i+1] = 0.5 * (t1[i] + t1[i+1])

    t2[2*N] = t1[N]
    return t2


def Cauchy_problem_error(F, t1, U0, scheme, q=None, **kwargs):

    # malla refinada
    t2 = refine_mesh(t1)

    # Soluciones
    U1 = Cauchy_problem(F, U0, t1, scheme, **kwargs)
    U2 = Cauchy_problem(F, U0, t2, scheme, **kwargs)

    N  = len(t1) - 1
    Nv = len(U0)

    Error = zeros((N+1, Nv))

    for i in range(N+1):
        Error[i,:] = U2[2*i,:] - U1[i,:]

    # Richardson
    if q is not None:
        Error = Error / (1 - 1/(2**q))

    return U1, Error

def Temporal_convergence_rate(F, U0, t, scheme, m=7, **kwargs):
    """
    Replica exacta del método del profesor, compatible con tu API.
    t es la malla inicial, NO el tiempo final.
    """

    N = len(t) - 1
    logE = zeros(m+1)
    logN = zeros(m+1)

    t1 = t.copy()

    for i in range(m+1):

        N = len(t1) - 1

        # Usa SIEMPRE Cauchy_problem_error 
        _, Error = Cauchy_problem_error(F, t1, U0, scheme, **kwargs)

        # error SOLO en tf (igual que el profe)
        logE[i] = log10(norm(Error[N,:]))
        logN[i] = log10(float(N))

        print("Error =", norm(Error[N,:]), " N =", N)

        # refinement EXACTO del profesor
        t1 = refine_mesh(t1)

    # El profe recorta valores demasiado pequeños
    y = logE[logE > -12]
    x = logN[0:len(y)]

    # Ajuste lineal logE = a logN + b
    order, b = polyfit(x, y, 1)

    print("order =", order, "b =", b)

    # Corrección Richardson final
    logE = logE - log10(1 - 1./2**abs(order))

    return logN, logE, order