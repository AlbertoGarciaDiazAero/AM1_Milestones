



## Imports

# Librerías estándar
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Librerías Científicas

# Visualización de grafos

# Procesado de imagen

# Clases propias necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) #Sube un nivel, donde está "Modules"
from Modules.temporal_schemes import Esquema, Cauchy_problem
from Modules.dynamic_functions import Kepler

def method_order(method):
    """
    Devuelve el orden teórico q del esquema temporal.
    method puede ser la función (Esquema['Euler'], RK4, etc.) o un string.
    """
    if isinstance(method, str):
        name = method
    else:
        name = getattr(method, "__name__", "")

    if name in ("Euler", "EulerInverso", "Inverse_Euler"):
        return 1
    elif name in ("Crank_Nicolson", "CrankNicolson"):
        return 2
    elif name == "RK4":
        return 4
    else:
        raise ValueError(f"No conozco el orden para el método '{name}'")

def richardson_error(F, U0, T, method, N, **kwargs):
    """
    Calcula el error por Richardson comparando soluciones con paso h y h/2.
    """
    # Integración con paso h
    t1 = np.linspace(0, T, N)
    U1 = Cauchy_problem(F, U0, t1, temporal_scheme=method, **kwargs)

    # Integración con paso h/2
    t2 = np.linspace(0, T, 2*N)
    U2 = Cauchy_problem(F, U0, t2, temporal_scheme=method, **kwargs)

        # Solución fina muestreada en los mismos tiempos
    U2_half = U2[::2, :]

    # Error base en el estado final
    base_err = norm(U1[-1] - U2_half[-1])

    # Orden del método y factor de Richardson
    q = method_order(method)
    return base_err / (1.0 - 0.5**q)

def convergence_rate_refinements(F, U0, T, temporal_scheme, N_ini, N_refinements=6, **kwargs):
    """
    Versión 'a lo profe': parte de N_ini y va refinando
    N -> 2N-1, comparando siempre la solución fina con la anterior.

    Devuelve:
        log_N      : log(N_fine) para cada refinamiento
        log_errors : log(error) asociado a cada N_fine
    """
    N = N_ini
    errors = np.zeros(N_refinements)
    log_N = np.zeros(N_refinements)

    # malla inicial (más gruesa)
    t_coarse = np.linspace(0.0, T, N)
    U1 = Cauchy_problem(F, U0, t_coarse,
                        temporal_scheme=temporal_scheme,
                        **kwargs)

    for i in range(N_refinements):
        # refinamos: paso ~ h/2
        N_fine = 2 * N - 1
        t_fine = np.linspace(0.0, T, N_fine)
        U2 = Cauchy_problem(F, U0, t_fine,
                            temporal_scheme=temporal_scheme,
                            **kwargs)

        # error: diferencia entre la solución nueva y la anterior
        errors[i] = norm(U2[-1, :] - U1[-1, :])
        log_N[i] = np.log(N_fine)

        # preparamos siguiente iteración
        U1 = U2
        N = N_fine

    log_errors = np.log(errors)
    return log_N, log_errors


# Condición inicial y parámetros generales
U0 = np.array([1.0, 0.0, 0.0, 1.0])

T = 10
N = 1000
t = np.linspace(0.0, T, N)
delta_t = t[1] - t[0]


methods = {
    "Euler": Esquema["Euler"],
    "EulerInverso": Esquema["Inverse_Euler"],
    "CrankNicolson": Esquema["Crank_Nicolson"],
    "RK4": Esquema["RK4"]
}
Ns_conv = np.array([200, 400, 800, 1600, 3200])

errors = {}
for name, scheme in methods.items():
    e = richardson_error(Kepler, U0, T, scheme, N,
                         jacobian_tol=1e-10, N_max=10000, newton_tol=1e-10)
    errors[name] = e

print("\nErrores Richardson:")
for m, e in errors.items():
    print(f"{m:15s}  Error = {e:.3e}")


############## Tasas de convergencia (refinamientos) ###################

print("\nTasas de convergencia (aprox, refinamientos):")

N_ini = 50         # N inicial 
N_ref = 5          # número de refinamientos sucesivos

for name, scheme in methods.items():
    # scheme es la función: Esquema["Euler"], Esquema["RK4"], etc.
    logN, logE = convergence_rate_refinements(
        Kepler,
        U0,
        T,
        temporal_scheme=scheme,
        N_ini=N_ini,
        N_refinements=N_ref,
        jacobian_tol=1e-9,
        N_max=10000,
        newton_tol=1e-9,
        verbose=False
    )

    # Ajuste lineal log(E) = a log(N) + b
    a, b = np.polyfit(logN, logE, 1)
    p_est = -a   # el orden es menos la pendiente

    print(f"{name:15s}  p ≈ {p_est:.2f}")


##############################
# Plot log(E) vs log(N) con refinamientos
##############################

plt.figure()

# N_ini grande para estar en régimen asintótico
N_ini = 2000
N_ref = 5

for name, scheme in methods.items():
    # scheme es la función: Esquema["Euler"], etc.
    logN, logE = convergence_rate_refinements(
        Kepler,
        U0,
        T,
        temporal_scheme=scheme,
        N_ini=N_ini,
        N_refinements=N_ref,
        jacobian_tol=1e-9,
        N_max=10000,
        newton_tol=1e-9,
        verbose=False
    )

    # ajuste lineal log(E) = a log(N) + b
    a, b = np.polyfit(logN, logE, 1)
    p = -a  # orden de convergencia teórico ≈ -pendiente

    # puntos + recta ajustada
    plt.scatter(logN, logE)
    plt.plot(logN, a * logN + b, label=f"{name} (p≈{p:.2f})")

plt.xlabel("log(N)")
plt.ylabel("log(E)")
plt.title("log(E) vs log(N) con refinamientos sucesivos")
plt.legend()
plt.tight_layout()
plt.show()