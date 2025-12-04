## Imports

# Librerías estándar
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Librerías Científicas

# Visualización de grafos

# Procesado de imagen

# Clases propias necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) #Sube un nivel, donde está "Modules"
from Modules.temporal_schemes import Esquema, Cauchy_problem
from Modules.dynamic_functions import Kepler, LinearOscillator, NBody


for k in Esquema.keys():
    print(k)

G = 1.0
masses = np.array([1.0, 1.1, 1.0])

# Condiciones iniciales (configuración triangular) ((Intento porque claramente están en orden incorrecto jajajaj))
U0 = np.array([
     1.0,  0.0,   0.0,  0.5,
    -0.5,  0.866, 0.0, -0.5,
    -0.5, -0.866, 0.0,  0.0
])

T = 3.6
N = 5000
t = np.linspace(0.0, T, N)



method = Esquema["Leap_Frog_Velocity_Verlet"]

U = Cauchy_problem(
    lambda U,t: NBody(U, t, masses, G),
    U0,
    t,
    scheme=method,
    jacobian_tol=1e-9,
    N_max=10000,
    newton_tol=1e-9,
    verbose=False
)


N_bodies = len(masses)
r = U[:,:2*N_bodies].reshape(N, N_bodies, 2)

plt.figure()
for i in range(N_bodies):
    plt.plot(r[:,i,0], r[:,i,1], label=f"Cuerpo {i+1}")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbitas del problema de 3 cuerpos")
plt.legend()
plt.grid(True)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(N_bodies):
    ax.plot(r[:, i, 0], r[:, i, 1], t, label=f"Cuerpo {i+1}")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("t")
ax.set_title("Órbitas N-cuerpos en (x, y, t)")
ax.legend()
plt.tight_layout()
plt.show()


def energia_nbody(U, masses, G=1.0):
    N = len(masses)
    U = np.array(U)
    r = U[:2*N].reshape(N, 2)
    v = U[2*N:].reshape(N, 2)

    # Energía cinética: 0.5 * sum(m_i * |v_i|^2)
    v2 = np.sum(v*v, axis=1)          # |v_i|^2
    K = 0.5 * np.sum(masses * v2)

    # Energía potencial: -G * sum_{i<j} m_i m_j / |r_i - r_j|
    V = 0.0
    for i in range(N):
        for j in range(i+1, N):
            V -= G * masses[i] * masses[j] / np.linalg.norm(r[j] - r[i])

    return K + V

# --- Cálculo y plot de la energía ---
E = np.array([energia_nbody(U[k], masses, G) for k in range(N)])

plt.figure()
plt.plot(t, E)
plt.xlabel("t")
plt.ylabel("E(t)")
plt.title("Energía total del sistema N-cuerpos")
plt.grid(True)
plt.show()
#Se ve la aproximación de los cuerpos 





###########################################################
####################### ¿PERIODICA? #######################
###########################################################
#La de lagrange tiene la putada de la aproximación, voy a ver si encuentro 

# --- Parámetros figure-eight 3 cuerpos ---
G = 1.0
masses = np.array([1.0, 1.21, 1.0])
N_bodies = len(masses)

# Condiciones iniciales  de Chenciner & Montgomery
x0  = 0.98000436
y0  = -0.24308753
vx0 = 0.4662036850
vy0 = 0.4323657300

# Posiciones
r1 = np.array([ x0,  y0])
r2 = np.array([-x0, -y0])
r3 = np.array([ 0.0, 0.0])

# Velocidades
v1 = np.array([ vx0,  vy0])
v2 = np.array([ vx0,  vy0])
v3 = np.array([-2*vx0, -2*vy0])

U0 = np.array([*r1, *r2, *r3, *v1, *v2, *v3])

# Periodo aproximado de la figure-eight
T_per = 6.3259
n_periods = 3
T = n_periods * T_per

N = 40000              # dt ~  T/N suficientemente pequeño
t = np.linspace(0.0, T, N)

method = Esquema["RK4"]   # o "RK4", pero Leap_Frog conserva mejor


U = Cauchy_problem(
    lambda U, tt: NBody(U, tt, masses, G),
    U0,
    t,
    scheme=method,
    jacobian_tol=1e-9,
    N_max=10000,
    newton_tol=1e-9,
    verbose=False
)

r = U[:, :2*N_bodies].reshape(N, N_bodies, 2)

plt.figure()
for i in range(N_bodies):
    plt.plot(r[:, i, 0], r[:, i, 1], label=f"Cuerpo {i+1}")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Órbita periódica figure-eight, {n_periods} periodos")
plt.legend()
plt.grid(True)
plt.show()

### ## ############### ## ###
#  #  #   Animación   #  #  #
### ## ############### ## ###

fig, ax = plt.subplots()
ax.set_aspect("equal", "box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Figure-eight 3 cuerpos ({n_periods} periodos)")
ax.grid(True)

# Límites
margin = 0.5
xmin = np.min(r[:, :, 0]) - margin
xmax = np.max(r[:, :, 0]) + margin
ymin = np.min(r[:, :, 1]) - margin
ymax = np.max(r[:, :, 1]) + margin
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

lines = []
points = []
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(N_bodies):
    # trayectoria
    line, = ax.plot([], [], lw=1.5, color=colors[i])
    # punto
    pt,   = ax.plot([], [], "o", color=colors[i])
    lines.append(line)
    points.append(pt)

# Para no usar los N pasos si son muchos
step = 10
frames = list(range(0, N, step))

def init():
    for line, pt in zip(lines, points):
        line.set_data([], [])
        pt.set_data([], [])
    return lines + points

def update(frame_idx):
    k = frame_idx  # ya es un índice de r porque frames = range(0,N,step)
    for i in range(N_bodies):
        # Trayectoria hasta el instante k (arrays -> OK)
        lines[i].set_data(r[:k+1, i, 0], r[:k+1, i, 1])
        # Posición instantánea: envolver en listas para que sean "sequences"
        points[i].set_data([r[k, i, 0]], [r[k, i, 1]])
    return lines + points

anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    interval=20,   # ms entre frames
    blit=True
)

plt.show()

E_Lagrange_RK4 = np.array([
    energia_nbody(U[k], masses, G) for k in range(N)
])

plt.figure()
plt.plot(t, E_Lagrange_RK4)
plt.xlabel("t")
plt.ylabel("E(t)")
plt.title("Energía total – caso Lagrange (3 cuerpos, RK4)")
plt.grid(True)
plt.show()

print("E(0)        =", E_Lagrange_RK4[0])
print("E min       =", E_Lagrange_RK4.min())
print("E max       =", E_Lagrange_RK4.max())
print("ΔE = max-min =", E_Lagrange_RK4.max() - E_Lagrange_RK4.min())
print("Error rel   =", (E_Lagrange_RK4.max() - E_Lagrange_RK4.min())
                       / abs(E_Lagrange_RK4.mean()))

## Apuntes del ticher (Mirar su github)
