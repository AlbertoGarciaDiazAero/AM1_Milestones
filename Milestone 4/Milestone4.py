
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
from Modules.temporal_schemes import Esquema, Cauchy_problem, Esquema_s
from Modules.dynamic_functions import Kepler, LinearOscillator
from Modules.Math import Stability_Region


#Condición inicial, generalidades, etc.
U0 = np.array([1.0, 0.0])   # x(0)=1, v(0)=0
T = 200.0                    # Subido a 200 para analizar cuestiones energéticas (Euler aumenta, Inverso decrece, . . .)
N = 10000
t = np.linspace(0.0, T, N)

methods = { #A ver se puede usar Esquemas, pongo esto por definir menos métodos o whatever
    "Euler":        Esquema["Euler"],
    "EulerInverso": Esquema["Inverse_Euler"],
    "LeapFrog":     Esquema["Leap_Frog"],
    "CrankNicolson":Esquema["Crank_Nicolson"],
    "RungeKutta4":          Esquema["RK4"],
}


sols = {}

for name, esquema in methods.items():
    U = Cauchy_problem(
        LinearOscillator,
        U0,
        t,
        scheme=esquema,
        jacobian_tol=1e-9,
        N_max=10000,
        newton_tol=1e-9,
        verbose=False
    )
    sols[name] = U

    # x(t)
    plt.figure()
    plt.plot(t, U[:,0])
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"Oscilador lineal con {name}")
    plt.grid(True)
    plt.show()

    # diagrama de fase (x vs v)
    plt.figure()
    plt.axis("equal")
    plt.plot(U[:,0], U[:,1])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(f"Diagrama de fase con {name}")
    plt.grid(True)
    plt.show()



############################
# Stability functions R(z). Añadido método general en Math.py
############################

def R_euler(z):
    return 1 + z

def R_inverse_euler(z):
    return 1.0 / (1.0 - z)

def R_crank_nicolson(z):
    return (1.0 + 0.5*z) / (1.0 - 0.5*z)

def R_rk4(z):
    return 1.0 + z + 0.5*z**2 + (1.0/6.0)*z**3 + (1.0/24.0)*z**4

def rho_leapfrog(z):
    """
    Devuelve rho(z) = max(|xi1|, |xi2|) para el esquema Leap-Frog:
        xi^2 - 2 z xi - 1 = 0
    """
    disc = np.sqrt(z**2 + 1.0)
    xi1 = z + disc
    xi2 = z - disc
    return np.maximum(np.abs(xi1), np.abs(xi2))


def plot_stability_region(method_name, xlim=(-4, 4), ylim=(-4, 4), npts=400, dt_mark=None):
    """
    Dibuja la región de estabilidad en el plano z = λ Δt para un método dado.
    method_name: 'Euler', 'Inverse_Euler', 'Crank_Nicolson', 'RK4', 'Leap_Frog'
    dt_mark: si no es None, marca en el eje imaginario z = i * dt_mark (oscilador lineal).
    """
    x = np.linspace(xlim[0], xlim[1], npts)
    y = np.linspace(ylim[0], ylim[1], npts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    if method_name == "Euler":
        R = R_euler(Z)
        rho = np.abs(R)
    elif method_name == "Inverse_Euler":
        R = R_inverse_euler(Z)
        rho = np.abs(R)
    elif method_name == "Crank_Nicolson":
        R = R_crank_nicolson(Z)
        rho = np.abs(R)
    elif method_name == "RK4":
        R = R_rk4(Z)
        rho = np.abs(R)
    elif method_name == "Leap_Frog":
        rho = rho_leapfrog(Z)
    else:
        raise ValueError(f"Método desconocido: {method_name}")

    stable = rho <= 1.0

    plt.figure()
    # Región estable (relleno)
    plt.contourf(X, Y, stable, levels=[0, 0.5, 1], alpha=0.4)
    # Frontera |R|=1
    plt.contour(X, Y, rho, levels=[1.0], colors='k', linewidths=1.0)

    # Ejes
    plt.axhline(0.0, color='black', linewidth=0.5)
    plt.axvline(0.0, color='black', linewidth=0.5)

    # Si quieres marcar el punto z = i Δt del oscilador (λ = ±i):
    if dt_mark is not None:
        plt.plot(0.0, dt_mark, 'rx', label=f"z = iΔt (Δt={dt_mark:.2g})")
        plt.plot(0.0, -dt_mark, 'rx')

    plt.gca().set_aspect('equal', 'box')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title(f"Región de estabilidad de {method_name}")
    if dt_mark is not None:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



T = 20.0
N = 200
t = np.linspace(0.0, T, N)
dt = t[1] - t[0]

metodos_stab = ["Euler", "Inverse_Euler", "Leap_Frog", "Crank_Nicolson", "RK4"]

for m in metodos_stab:
    plot_stability_region(m, xlim=(-4, 4), ylim=(-4, 4), npts=400, dt_mark=dt)


## Regiones de estabilidad mediante la ecuación característica. Función "Stability_Region" de Modules.Math
## es como resuelve J.A.
## Estoy trabajando en ello todvía, poco a poco


for nombre, scheme in Esquema_s.items():
    print("Región de estabilidad:", nombre)
    rho, x, y = Stability_Region(scheme, 150, -4, 4, -4, 4)
    plt.contour(x, y, rho.T, levels=np.linspace(0,1,11))
    plt.axis("equal")
    plt.grid()
    plt.show()