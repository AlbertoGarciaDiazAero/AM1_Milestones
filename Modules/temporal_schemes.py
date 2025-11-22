'''
No sé si me convence tanta modularización, pero ueno. 

Definición de esquemas temporales:
    
    - Euler
    - CN
    - RK4
    - Euler Inverso
    -

Definción de diccionario de esquemas-> Limpia, fija y da esplendor

Problema de Cauchy->Función mandadora
'''

from numpy import array, concatenate, zeros, linspace
from numpy.linalg import norm, solve, LinAlgError
import matplotlib.pyplot as plt

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

def Crank_Nicolson(F, U, ti, tf, N_max=None, newton_tol=None, jacobian_tol=None, verbose=False, **kwargs):
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
    verbose: Aporta información de convergencias
    """
    def crank_nicolson_target_function(U_n_plus_1, F, U_n, ti, tf):
        delta_t = tf-ti
        return (0.5*(F(U_n_plus_1, tf) + F(U_n, ti))) - ((U_n_plus_1 - U_n)/delta_t)
    
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

    def newton(f, x, h, N, newton_tol, F, U_n, ti, tf):

        x_n = array(x, copy=True)
        for n in range(0, N):
            J_n = Jacobian(f, x_n, h, F, U_n, ti, tf)
            f_n = f(x_n, F, U_n, ti, tf)
            try:
                delta_x = solve(J_n, -f_n)
            except LinAlgError:
                verbose and print("LinAlgError: Newton-Raphson ill-defined.")
                break
            
            x_n = x_n + delta_x

            if(norm(delta_x)<newton_tol):
                verbose and print(f"Alcanzado newton en {n} iteraciones ") #Printea solo si verbose=True
                break

        return x_n

    sol = newton(crank_nicolson_target_function, U, jacobian_tol, N_max, newton_tol, F, U, ti, tf)

    return sol

def RK4(F,U, ti, tf, **kwargs):
    '''
    
    '''
    k1 = F(U, ti)
    k2 = F(U+0.5*k1*(tf-ti), ti+0.5*(tf-ti))
    k3 = F(U+0.5*k2*(tf-ti), ti+0.5*(tf-ti))
    k4 = F(U+k3*(tf-ti), ti+(tf-ti))

    return U + (1.0/6.0)*(tf-ti)*(k1 + 2*k2+2*k3 + k4)

def Inverse_Euler(F, U, ti, tf, N_max=None, newton_tol=None, jacobian_tol=None, verbose=False, **kwargs):

    """
    
    """
    def inverse_euler_target_function(U_n_plus_1, F, U_n, ti, tf):
        return U_n_plus_1-U_n-(tf-ti)*F(U_n_plus_1, tf)

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

    def newton(f, x, h, N, newton_tol, F, U_n, ti, tf):

        x_n = array(x, copy=True)
        for n in range(0, N):
            J_n = Jacobian(f, x_n, h, F, U_n, ti, tf)
            f_n = f(x_n, F, U_n, ti, tf)
            try:
                delta_x = solve(J_n, -f_n)
            except LinAlgError:
                verbose and print("LinAlgError: Newton-Raphson ill-defined.")
                break
            
            x_n = x_n + delta_x

            if(norm(delta_x)<newton_tol):
                verbose and print(f"Alcanzado newton en {n} iteraciones ") #Printea solo si verbose=True
                break

        return x_n
    
    sol = newton(inverse_euler_target_function, U, jacobian_tol, N_max, newton_tol, F, U, ti, tf)

    return sol

def Leap_Frog(F, U, ti, tf, **kwargs):
    """
    Leap-Frog / Velocity-Verlet para sistemas de la forma:
        U = [r, v],  F(U,t) = [v, a(r,t)]
    (válido para el oscilador lineal y otros sistemas x¨ = a(x).)

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

#################################
#### Diccionario de Esquemas ####
#################################

Esquema = {
    "Euler": Euler,
    "RK4": RK4,
    "Crank_Nicolson": Crank_Nicolson,
    "Inverse_Euler": Inverse_Euler,
    "Leap_Frog": Leap_Frog,    
}
#################################
###### Problema de Cauchy #######
#################################

#Mirar Milestone2 para ejemplo de uso, que ahí lo dejo solo con estas funciones.
def Cauchy_problem(F, U0, t,
                   temporal_scheme="Euler",
                   N_max=None, newton_tol=None, jacobian_tol=None,
                   verbose=False):
    """
    temporal_scheme: string con el nombre del esquema ("Euler", "RK4", ...)
    """
    if isinstance(temporal_scheme, str):
        try:
            scheme_func = Esquema[temporal_scheme]
        except KeyError:
            raise ValueError(f"Esquema temporal desconocido: {temporal_scheme}")
    else:
        # por si quieres seguir pasando directamente la función
        scheme_func = temporal_scheme

    N = len(t)
    N_v = len(U0)
    U = zeros((N, N_v))
    U[0, :] = U0

    for n in range(0, N-1):
        U[n+1, :] = scheme_func(
            F, U[n, :], t[n], t[n+1],
            N_max=N_max,
            newton_tol=newton_tol,
            jacobian_tol=jacobian_tol,
            verbose=verbose,
        )

    return U

