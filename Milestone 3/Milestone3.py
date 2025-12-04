'''
Docstring for Milestone 3.Milestone3


'''




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
from Modules.dynamic_functions import Kepler, LinearOscillator
from Modules.Errors import Temporal_convergence_rate,  Cauchy_problem_error


###########################
# Parámetros del problema #
###########################

U0 = np.array([1.0, 0.0, 0.0, 1.0])   # Estado inicial (orbita circular)
U0 = np.array([1.0,0.0])              # Estado inicial edl oscilador 

T  = 30                             # Tiempo final
N  = 10000                              # Puntos iniciales para las pruebas de error

# Métodos a evaluar
methods = {
    "Euler"          : Esquema["Euler"],
    # "Inverse_Euler"  : Esquema["Inverse_Euler"], #Me gusta mi vida a veces. Vamos a quitar a este badboy que si no el tiempo de compilación lo disparamos.
    "Crank_Nicolson" : Esquema["Crank_Nicolson"],
    "RK4"            : Esquema["RK4"],
    "Leap_Frog_VV"   : Esquema["Leap_Frog_Velocity_Verlet"]
}

##########################
# 1. ERROR DE RICHARDSON #
##########################

print("\n=== ERROR DE RICHARDSON ===\n")

for name, scheme in methods.items():
    # obtener orden teórico q
    if name == "Euler":
        q = 1
    elif name in ("Inverse_Euler", "Crank_Nicolson"):
        q = 1 if name == "Inverse_Euler" else 2
    elif name == "RK4":
        q = 4

    t = np.linspace(0, T, N)

    U1, Error = Cauchy_problem_error(
        #F=Kepler,
        F=LinearOscillator,
        t1=t,
        U0=U0,
        scheme=scheme,
        q=q
    )

    E=Error
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ############################
    # 1) plot
    ############################
    axes[0].plot(t, E, ".-")
    axes[0].set_title(f"Error Richardson - {name}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("||Error||")
    axes[0].grid(True)

    ############################
    # 2) PLOT 2
    ############################
    axes[1].plot(t, np.abs(Error[:, 1]))
    axes[1].set_yscale("log")
    axes[1].set_title(f"{name} error over time (abs) (log scale)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("log(Error))")
    axes[1].grid(True)
    axes[1].set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()


###########################################
# 2. TASA DE CONVERGENCIA (refinamientos) #
###########################################

print("\n=== TASAS DE CONVERGENCIA ===\n")

N_ini       = 100     # malla inicial pequeña
refinements = 13      # nº de refinamientos N → 2N−1

for name, scheme in methods.items():

    logN, logE, order  = Temporal_convergence_rate(
        #F=Kepler,
        F=LinearOscillator, #Mejor en este el LinearOscillator, va más rápidin y para altos refinamientos se nota.
        U0=U0,
        t=np.linspace(0, T, N_ini),
        scheme=scheme,
        N_ini=N_ini,
        m=refinements
    )

    # Ajuste lineal: logE = a logN + b  →  orden = -a
    a, b = np.polyfit(logN, logE, 1)
    p = -a

    print(f"{name:15s}  orden ≈ {p:.2f}")

    plt.figure()
    plt.scatter(logN, logE)
    plt.plot(logN, a*logN + b, label=f"{name} (p≈{p:.2f})")
    plt.xlabel("log(N)")
    plt.ylabel("log(error)")
    plt.legend()
    plt.grid(True)
    plt.show()

