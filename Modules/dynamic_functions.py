'''
Guardo aquí algunas F(U) de problemas dinámicos conocidos, el ppal es Kepler en esta asignatura pero vaya
'''

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



def Kepler(U, t=0):
    """
    Returns F(U)=(\\dot{r}, -r/|r|^3), where r is the position vector.
    Note: F is not explicitly dependent on t in this case, adding it allows to construct
    robust functions for the numerical methods.
    """
    r = U[0:2]
    v = U[2:4]

    return np.concatenate((v, -r/np.linalg.norm(r)**3), axis=None)

def LinearOscillator(U, t=0):
    """
    Sistema de 1er orden equivalente a x¨ + x = 0:
        U = [x, v],  v = x'
    """
    x = U[0]
    v = U[1]
    return np.array([v, -x])


def NBody(U, t, masses, G=1.0):
    """
    Gravitational N-body problem in 2D.
    U = [x1, y1, vx1, vy1, ..., xN, yN, vxN, vyN]
    masses: array-like of length N
    G: gravitational constant (scaled by convenience)
    """
    U = np.array(U, dtype=float)
    N = len(masses)

    # Separar posiciones y velocidades
    r = U[:2*N].reshape(N, 2)     # (N,2)
    v = U[2*N:].reshape(N, 2)     # (N,2)

    # Aceleraciones
    a = np.zeros((N,2), dtype=float)

    for i in range(N):
        for j in range(N):
            if i != j:
                rij = r[j] - r[i]
                dist = np.linalg.norm(rij)
                a[i] += G * masses[j] * rij / dist**3

    # Ensamblar derivada U' = (v, a)
    dUdt = np.concatenate((v.flatten(), a.flatten()))
    return dUdt