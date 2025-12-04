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

def oscillator(U, t):
    x, y = U
    return np.array([ y, -x ])


def NBody(U, t, masses, G=1.0):
    """
    Gravitational N-body problem in 2D.
    U = [x1, y1, x2, y2, ..., vx1, vy1, ..., vxN, vyN]
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



def N_body2(U, t, Nb, Nc, Nv=2):
    # U: vector plano de tamaño Nb*Nc*Nv

    # ----- Vistas sobre U -----
    Xs = U.reshape(Nb, Nc, Nv)   # (Nb, Nc, 2)
    r  = Xs[:, :, 0]             # posiciones (Nb, Nc)  -> vista
    v  = Xs[:, :, 1]             # velocidades (Nb, Nc) -> vista

    # ----- Creamos F y sus vistas -----
    F  = np.zeros_like(U)        # derivada dU/dt plana
    Fs = F.reshape(Nb, Nc, Nv)   # (Nb, Nc, 2)
    drdt = Fs[:, :, 0]           # vista dentro de F
    dvdt = Fs[:, :, 1]           # vista dentro de F

    # ----- Ecuaciones -----
    for i in range(Nb):
        # dr/dt = v
        drdt[i, :] = v[i, :]

        # dv/dt = suma de atracciones
        for j in range(Nb):
            if i != j:
                rij  = r[j, :] - r[i, :]
                dist = np.linalg.norm(rij)
                dvdt[i, :] += rij / dist**3

    return F


def restricted_3body_F(U, t, mu):
    """
    Circular Restricted Three Body Problem (planar, normalizado).
    U = [x, y, vx, vy] en un marco rotante.
    mu = m2 / (m1 + m2)
    """

    U = np.asarray(U, dtype=float)

    # Vista 2x2 sobre U: eje 0 -> coord (x,y), eje 1 -> (pos, vel)
    Xs = U.reshape(2, 2)     # [[x, vx],
                             #  [y, vy]]
    r  = Xs[:, 0]            # [x, y]
    v  = Xs[:, 1]            # [vx, vy]

    # Derivadas (mismas vistas sobre F)
    F  = np.zeros_like(U)
    Fs = F.reshape(2, 2)
    drdt = Fs[:, 0]          # dx/dt, dy/dt
    dvdt = Fs[:, 1]          # dvx/dt, dvy/dt

    x, y   = r
    vx, vy = v

    # Distancias a los primarios en el sistema normalizado
    r1 = np.sqrt((x + mu)**2       + y**2)
    r2 = np.sqrt((x - 1 + mu)**2   + y**2)

    # Ecuaciones del CR3BP en marco rotante
    drdt[0] = vx
    drdt[1] = vy

    dvdt[0] = 2*vy + x - ((1-mu)*(x + mu))/r1**3 - (mu*(x - (1-mu)))/r2**3
    dvdt[1] = -2*vx + y - ((1-mu)*y)/r1**3       - (mu*y)/r2**3

    return F