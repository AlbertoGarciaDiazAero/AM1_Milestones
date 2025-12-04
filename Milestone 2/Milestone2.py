'''
  Docstring de manual.

  Me he llevado todas las funciones que deberían ser este módulo a temporal_schemes. Para tener mayor trazabilidad y solo modificar
  las cosas en un archivo, saludos.

  Alberto García Díaz
'''
# Conda Env Usado: TFG-min. Python 3.10.18 
# Requirements: numpy, mathplotlib. Modules


## Imports

# Librerías estándar
import numpy as np
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


## Generalidades

def Kepler(U, t=0):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    Note: F is not explicitly dependent on t in this case, adding it allows to construct
    robust functions for the numerical methods.
    """
    r = U[0:2]
    v = U[2:4]

    return np.concatenate((v, -r/np.linalg.norm(r)**3), axis=None)

def plot_orbit(U, method, N, T, delta_t):
    plt.figure()
    plt.axis("equal")
    plt.plot(U[:, 0], U[:, 1], label=f"N={N}, Δt={delta_t:.2e}")
    plt.scatter(U[:, 0], U[:, 1], color='teal', s=2, label="Time steps")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Órbita con {method}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# A resolver uiiiii

# Condición inicial
U0 = np.array([1.0, 0.0, 0.0, 1.0])

T = 100.0
N = 2000
t = np.linspace(0.0, T, N)
delta_t = t[1] - t[0]


# Diccionario de métodos a plottear:
metodos = {
    "Euler":        Esquema["Euler"],
    "RK4":          Esquema["RK4"],
    "CN":           Esquema["Crank_Nicolson"],
    "EulerInverso": Esquema["Inverse_Euler"],
    "Leap_Frog_Velocity_Verlet": Esquema["Leap_Frog_Velocity_Verlet"],
    "Leap_Frog": Esquema["Leap_Frog"],   
}

soluciones = {}

# Integrar con cada método
for nombre, scheme in metodos.items():
    U = Cauchy_problem(
        Kepler,
        U0,
        t,
        temporal_scheme=scheme,
        jacobian_tol=1e-10,
        N_max=10000,
        newton_tol=1e-10,
        verbose=False
    )
    soluciones[nombre] = U
    # Plot individual
    plot_orbit(U, nombre, N, T, delta_t)


#Plot Conjunto

plt.figure()
plt.axis("equal")

for nombre, U in soluciones.items():
    plt.plot(U[:, 0], U[:, 1], label=nombre)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de esquemas temporales")
plt.legend()
plt.tight_layout()
plt.show()

################
# Euler inverso para distintos N
################

Ns = [200, 500, 1000, 2000,3000,4000,5000]  
plt.figure()
plt.axis("equal")

for N_i in Ns:
    t_i = np.linspace(0.0, T, N_i)
    delta_t_i = t_i[1] - t_i[0]

    U_i = Cauchy_problem(
        Kepler,
        U0,
        t_i,
        temporal_scheme=Esquema["Inverse_Euler"],
        jacobian_tol=1e-10,
        N_max=10000,
        newton_tol=1e-10,
        verbose=False
    )

    plt.plot(U_i[:, 0], U_i[:, 1],
             label=f"N={N_i}, Δt={delta_t_i:.2e}")

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita con Euler inverso para distintos N")
plt.legend()
plt.tight_layout()
plt.show()