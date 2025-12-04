'''
Docstring for Milestone 6.Milestone6
'''
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) #Sube un nivel, donde está "Modules"
from Modules.temporal_schemes import RangeKutta45
from Modules.dynamic_functions import restricted_3body_F



# Parámetros físicos Tierra–Luna
M1 = 5.972e24
M2 = 7.348e22
mu = M2 / (M1 + M2)

# Estado inicial: [x, y, vx, vy]
U0 = np.array([0.85, 0.0, 0.01, 0.05], dtype=float) #Variar energía cinética para ver si acaba llegando a superar el pozo de potencial. (Minilunas o así)

# Intervalo de integración
t0 = 0.0
tf = 60.0
dt = 0.001
N  = int((tf - t0) / dt)

t = np.linspace(t0, tf, N + 1)
U_hist = np.zeros((N + 1, len(U0)))
U_hist[0, :] = U0

# F_local sólo depende de U porque RangeKutta45 espera F(U)
# Metemos mu por clausura y fijamos t (no depende explícitamente del tiempo)
def F_local(U):
    return restricted_3body_F(U, 0.0, mu)

# Bucle de integración con RK45 embebido de un paso
for n in range(N):
    U_hist[n + 1, :] = RangeKutta45(F_local, U_hist[n, :], t[n], t[n + 1])

# Para comodidad, lo llamo como en tu ejemplo
r_sat1 = U_hist      # columnas 0,1 = posición; 2,3 = velocidad
t1     = t

# Posiciones de Tierra y Luna en el sistema rotante normalizado
pos_tierra1 = np.array([-mu, 0.0])
pos_luna1   = np.array([1.0 - mu, 0.0])

# ----- PLOT -----
plt.figure()

plt.plot(r_sat1[:, 0], r_sat1[:, 1], "red", label='Trayectoria del satélite')
plt.plot(pos_tierra1[0], pos_tierra1[1], 'bo', markersize=10, label='Tierra')
plt.plot(pos_luna1[0],   pos_luna1[1],   'go', markersize=8,  label='Luna')
plt.plot(U0[0], U0[1], 'ro', markersize=5, label='Satélite')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Restricted 3-body problem (CR3BP)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

############################################################

x_L4 = 0.5 - mu
y_L4 = np.sqrt(3)/2

U0_L4 = np.array([
    x_L4 + 0.0001,     # x
    y_L4 + 0.0001,     # y
    0.0,             # vx
    0.0              # vy
])
t0 = 0.0
tf = 10.0
dt = 0.001
N  = int((tf - t0) / dt)

t = np.linspace(t0, tf, N + 1)
U_hist = np.zeros((N + 1, len(U0_L4)))
U_hist[0, :] = U0_L4




# Bucle de integración con RK45 embebido de un paso
for n in range(N):
    U_hist[n + 1, :] = RangeKutta45(F_local, U_hist[n, :], t[n], t[n + 1])

# Para comodidad, lo llamo como en tu ejemplo
r_sat1 = U_hist      # columnas 0,1 = posición; 2,3 = velocidad
t1     = t

# Posiciones de Tierra y Luna en el sistema rotante normalizado
pos_tierra1 = np.array([-mu, 0.0])
pos_luna1   = np.array([1.0 - mu, 0.0])

# ----- PLOT -----
plt.figure()

plt.plot(r_sat1[:, 0], r_sat1[:, 1], "red", label='Trayectoria del satélite')
plt.plot(pos_tierra1[0], pos_tierra1[1], 'bo', markersize=10, label='Tierra')
plt.plot(pos_luna1[0],   pos_luna1[1],   'go', markersize=8,  label='Luna')
plt.plot(U0_L4[0], U0_L4[1], 'ro', markersize=5, label='Satélite')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Restricted 3-body problem,entorno a L4')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()








x_L2 = 1 - mu + (mu/3)**(1/3)
y_L2 = 0.0

U0_L2 = np.array([
    x_L2 - 0.002,  # pequeña desviación hacia dentro
    0.0,
    0.0,
    0.08          # velocidad transversal que genera libración
])

t0 = 0.0
tf = 10.0
dt = 0.001
N  = int((tf - t0) / dt)

t = np.linspace(t0, tf, N + 1)
U_hist = np.zeros((N + 1, len(U0_L2)))
U_hist[0, :] = U0_L2

# Bucle de integración con RK45 embebido de un paso
for n in range(N):
    U_hist[n + 1, :] = RangeKutta45(F_local, U_hist[n, :], t[n], t[n + 1])

# Para comodidad, lo llamo como en tu ejemplo
r_sat1 = U_hist      # columnas 0,1 = posición; 2,3 = velocidad
t1     = t

# Posiciones de Tierra y Luna en el sistema rotante normalizado
pos_tierra1 = np.array([-mu, 0.0])
pos_luna1   = np.array([1.0 - mu, 0.0])

# ----- PLOT -----
plt.figure()

plt.plot(r_sat1[:, 0], r_sat1[:, 1], "red", label='Trayectoria del satélite')
plt.plot(pos_tierra1[0], pos_tierra1[1], 'bo', markersize=10, label='Tierra')
plt.plot(pos_luna1[0],   pos_luna1[1],   'go', markersize=8,  label='Luna')
plt.plot(U0_L2[0], U0_L2[1], 'ro', markersize=5, label='Satélite')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Restricted 3-body problem,entorno a L2')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
