'''
No sé si me convence tanta modularización, pero ueno. 

Definición de esquemas temporales:
    
    - Euler
    - CN
    - RK4
    - Euler Inverso

Dos tipos: Simples (_s) y "los otros". 
Los primeros son los de J.A., los otros tienen cositas como Newton de Math.py, en esencia idénticos.

Problema de Cauchy->Función mandadora
'''

from numpy import array, concatenate, zeros, linspace
from numpy.linalg import norm, solve, LinAlgError
import matplotlib.pyplot as plt
from scipy.optimize import newton

from Modules.Math import Jacobian, Newton 


################################################################
####################### Esquemas Simples #######################
################################################################



def Euler_s(U, dt, t, F):
    return U + dt * F(U, t)


def Inverse_Euler_s(U, dt, t, F):
    def Residual(X):
        return X - U - dt * F(X, t)

    return newton(func=Residual, x0=U)


def Crank_Nicolson_s(U, dt, t, F):
    def Residual_CN(X):
        return X - a - (dt/2) * F(X, t + dt)

    a = U + (dt/2) * F(U, t)
    return newton(Residual_CN, U)


def RK4_s(U, dt, t, F):
    k1 = F(U, t)
    k2 = F(U + dt*k1/2, t + dt/2)
    k3 = F(U + dt*k2/2, t + dt/2)
    k4 = F(U + dt*k3,   t + dt)

    return U + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def Leap_Frog_s(U, U_prev, dt, t, F):
    return U_prev + 2.0*dt*F(U, t)

def Leap_Frog_VV_s(U, dt, t, F):
    h = dt
    dim = len(U) // 2

    r_n = U[:dim]
    v_n = U[dim:]

    # Evaluar aceleración en t_n
    k_n = F(U, t)
    a_n = k_n[dim:]

    # Paso 1: velocidad a mitad de paso
    v_half = v_n + 0.5*h*a_n

    # Paso 2: nueva posición
    r_np1 = r_n + h*v_half

    # Paso 3: aceleración en t_{n+1}
    U_star = concatenate((r_np1, v_half))
    k_np1 = F(U_star, t + h)
    a_np1 = k_np1[dim:]

    # Paso 4: velocidad en t_{n+1}
    v_np1 = v_half + 0.5*h*a_np1

    return concatenate((r_np1, v_np1))

Esquema_s = {
    "Euler"                         : Euler_s,
    "Inverse_Euler"                 : Inverse_Euler_s,
    "Crank_Nicolson"                : Crank_Nicolson_s,
    "RK4"                           : RK4_s,
}


#################################
###### Esquemas temporales ######
#################################

def Euler(F, U, ti, tf, **kwargs):
    """
    Classic Euler. Devuelve U n+1

    params
    -------
    F: Función F(U,t) característica del problema.
    U: Variable de estado.
    ti: Tiempo inicial.
    tf: Tiempo Final.

    """
    return U + (tf-ti)*F(U, ti)

def Crank_Nicolson(F, U, ti, tf, N_max=20, newton_tol=1e-7, jacobian_tol=1e-6, verbose=False, **kwargs):
    """
    Crank-Nicolson

    params
    -------
    F: Función F(U,t) característica del problema.
    U: Variable de estado.
    ti: Tiempo inicial.
    tf: Tiempo final.
    N_max: Número máximo de iteraciones Newton.
    newton_tol: Tolerancia del método de Newton
    jacobian_tol: Tolerancia del Jacobiano
    verbose: Printea información de convergencias
    """
    def crank_nicolson_target_function(U_n_plus_1, F, U_n, ti, tf):
        delta_t = tf-ti
        return (0.5*(F(U_n_plus_1, tf) + F(U_n, ti))) - ((U_n_plus_1 - U_n)/delta_t)

    sol = Newton(f=crank_nicolson_target_function, x=U, h=jacobian_tol, N=N_max, newton_tol=newton_tol, F=F, U_n=U, ti=ti, tf=tf, verbose=verbose)

    return sol

def RK4(F,U, ti, tf, **kwargs):
    '''
    Runge Kutta de Orden 4.

    params
    -------
    F: Función F(U,t) característica del problema.
    U: Variable de estado.
    ti: Tiempo inicial.
    tf: Tiempo Final.
    '''
    k1 = F(U, ti)
    k2 = F(U+0.5*k1*(tf-ti), ti+0.5*(tf-ti))
    k3 = F(U+0.5*k2*(tf-ti), ti+0.5*(tf-ti))
    k4 = F(U+k3*(tf-ti), ti+(tf-ti))

    return U + (1.0/6.0)*(tf-ti)*(k1 + 2*k2+2*k3 + k4)

def Inverse_Euler(F, U, ti, tf, N_max=20, newton_tol=1e-7, jacobian_tol=1e-6, verbose=False, **kwargs):

    """
    Euler Inverso. 
    params
    -------
    F: Función F(U,t) característica del problema.
    U: Variable de estado.
    ti: Tiempo inicial.
    tf: Tiempo final.
    N_max: Número máximo de iteraciones Newton.
    newton_tol: Tolerancia del método de Newton
    jacobian_tol: Tolerancia del Jacobiano
    verbose: Printea información de convergencias
    """
    def inverse_euler_target_function(U_n_plus_1, F, U_n, ti, tf):
        return U_n_plus_1-U_n-(tf-ti)*F(U_n_plus_1, tf)

    sol=Newton(f=inverse_euler_target_function, x=U, h=jacobian_tol, N=N_max, newton_tol=newton_tol, F=F, U_n=U, ti=ti, tf=tf, verbose=verbose)
    
    return sol

def Leap_Frog(F, U, t_prev, t_curr, U_prev, **kwargs):
    """
    Leap-Frog de dos pasos:
        U_{n+1} = U_{n-1} + 2 dt F(U_n, t_n)
    donde:
        U      ≡ U_n
        U_prev ≡ U_{n-1}
        t_prev ≡ t_{n-1}
        t_curr ≡ t_n
    """
    dt = t_curr - t_prev
    t_n = t_curr
    return U_prev + 2*dt*F(U, t_n)

def Leap_Frog_VV(F, U, ti, tf, **kwargs):
    """
    Leap-Frog / Velocity-Verlet para sistemas de la forma:
        U = [r, v],  F(U,t) = [v, a(r,t)]
    (válido para el oscilador lineal y otros sistemas x¨ = a(x).)
    Queda a programar el Leap_frog de veritá, que permita escoger esquema para arranque.
    U: [x, v]
    """
    h = tf - ti
    dim = len(U) // 2

    # Separar posición y velocidad
    r_n = U[:dim]
    v_n = U[dim:]

    # Evaluar aceleración en tn
    k_n = F(U, ti)
    a_n = k_n[dim:]

    # Paso 1: velocidad a mitad de paso
    v_half = v_n + 0.5 * h * a_n

    # Paso 2: nueva posición
    r_np1 = r_n + h * v_half

    # Para la aceleración en tn+1 necesitamos un estado provisional
    U_star = concatenate((r_np1, v_half))
    k_np1 = F(U_star, tf)
    a_np1 = k_np1[dim:]

    # Paso 3: velocidad en tn+1
    v_np1 = v_half + 0.5 * h * a_np1

    return concatenate((r_np1, v_np1))

def RangeKutta45(F, U,t1,t2, **kwargs):

        dt = t2 - t1

        k1 = dt*F(U)
        k2 = dt*F(U + (2/9)*k1)
        k3 = dt*F(U + (1/12)*k1 + (1/4)*k2)
        k4 = dt*F(U + (69/128)*k1 + (-243/128)*k2 + (135/64)*k3)
        k5 = dt*F(U + (-17/12)*k1 + (27/4)*k2 + (-27/5)*k3 + (16/15)*k4)
        k6 = dt*F(U + (65/432)*k1 + (-5/16)*k2 + (13/16)*k3 + (4/27)*k4 + (5/144)*k5)
        
        U1 = U + (47/450)*k1 + (0)*k2 + (12/25)*k3 + (32/225)*k4 + (1/30)*k5 + (6/25)*k6

        return U1


#################################
#### Diccionario de Esquemas ####
#################################

Esquema = {
    "Euler"                     : Euler,
    "RK4"                       : RK4,
    "Crank_Nicolson"            : Crank_Nicolson,
    "Inverse_Euler"             : Inverse_Euler,
    "Leap_Frog_Velocity_Verlet" : Leap_Frog_VV,
    "Leap_Frog"                 : Leap_Frog,    
}

#################################
###### Problema de Cauchy #######
#################################

#Mirar Milestone2 para ejemplo de uso, que ahí lo dejo solo con estas funciones.
def Cauchy_problem(F, U0, t, scheme, U1=None, **kwargs):
    """
    Resuelve dU/dt = F(U,t), U(t0)=U0
    usando un esquema temporal dado por `scheme`.

    Parámetros
    ----------
    F      : callable, F(U,t)
    U0     : array-like, estado inicial
    t      : array-like, malla temporal (t[0]...t[-1])
    scheme : callable
             - para TODOS salvo Leap_Frog: scheme(F, U_n, t_n, t_{n+1}, **kwargs)
             - para Leap_Frog:            Leap_Frog(F, U_n, t_{n-1}, t_n, U_{n-1}, **kwargs)
    U1     : opcional, estado en t[1] si quieres arrancar Leap_Frog con algo distinto de Euler

    Devuelve
    --------
    U : ndarray de shape (N_t, N_v)
    """

    Nt = len(t) - 1      # nº intervalos
    Nv = len(U0)         # nº variables
    U = zeros((Nt+1, Nv))
    U[0, :] = U0

    # Caso especial Leap-Frog (2-step)
    if scheme is Leap_Frog:
        if Nt == 0:
            return U

        # Si no se da U1, arrancamos con Euler
        if U1 is None:
            U[1, :] = Euler(F, U[0, :], t[0], t[1], **kwargs)
        else:
            U[1, :] = U1

        # Leap-Frog: U_{n+1} = U_{n-1} + 2*dt*F(U_n, t_n)
        for n in range(1, Nt):
            U[n+1, :] = Leap_Frog(
                F,
                U[n, :],      # U_n
                t[n-1],       # t_{n-1}
                t[n],         # t_n
                U[n-1, :],    # U_{n-1}
                **kwargs,
            )

    else:
        # Todos los demás esquemas son 1-step: U_{n+1} = scheme(F, U_n, t_n, t_{n+1})
        for n in range(Nt):
            U[n+1, :] = scheme(
                F,
                U[n, :],
                t[n],
                t[n+1],
                **kwargs,
            )

    return U

