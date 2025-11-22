'''
  Resolution of the Kepler Problem with FP Paradigm.
  1. Euler
      1. a) Euler con funciones
      1. b) Euler sin funciones

  2. Crank-Nicolson
  3. Runge Kutta - 4 
  4. Comparación esquemas
  
  Voy a procurar seguir Pep-8. Por costumbre también.

  Alberto García Díaz
'''
# Conda Env Usado: TFG-min. Python 3.10.18 



## Imports

# Librerías estándar
import numpy as np
import matplotlib.pyplot as plt

# Librerías Científicas

# Visualización de grafos

# Procesado de imagen


## Generalidades


# Definición de F(U):
def F(U):
        
            r = U[0:2]
            rd= U[2:4]

            return np.concatenate([rd, -r/np.linalg.norm(r)**3])


# Parámetros CN
T = 100
N = 1000


Delta_t = T/N

## Problema 
# CI
U0 = np.array([1, 0, 0, 1])


U_solucion = np.zeros((N+1, 4))

U_solucion[0,:] = U0

#####################
###     Euler     ###
#####################

#Esquema
for n in range(N):

 U_solucion[n+1, :] = U_solucion[n, :]  + Delta_t*F(U_solucion[n, :])


## Visualización Resultados

# print(U_solucion) (Ver la evolución numérica) 

# Gráfica
# Puntos x e y:
x_euler = U_solucion[:, 0]
y_euler = U_solucion[:, 1]


plt.plot(x_euler, y_euler) 
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita al problema de Kepler con el método de Euler. 1.a)")
plt.axhline(0, color='black', linewidth=0.8)  # eje x en y=0
plt.axvline(0, color='black', linewidth=0.8)  # eje y en x=0
plt.axis('equal')
plt.show()


###################################
###     Euler sin funciones     ###
################################### 

#Esquema integrado en bucle (Se usan las Generalidades de antes)
for n in range(N):
    
   rx, ry, vx, vy = U_solucion[n, :]

   r = np.sqrt(rx**2 + ry**2)

   U_solucion[n+1, 0] = rx + Delta_t * vx
   U_solucion[n+1, 1] = ry + Delta_t * vy
   U_solucion[n+1, 2] = vx + Delta_t * (-rx/ r**3)
   U_solucion[n+1, 3] = vy + Delta_t * (-ry/ r**3)

## Gráfica
x_euler = U_solucion[:, 0]
y_euler = U_solucion[:, 1]


plt.plot(x_euler, y_euler) 
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita al problema de Kepler con el método de Euler. 1.b)")
plt.axhline(0, color='black', linewidth=0.8)  # eje x en y=0
plt.axvline(0, color='black', linewidth=0.8)  # eje y en x=0
plt.axis('equal')
plt.show()


##############################
###     Crank-Nicolson     ###
##############################

## Generalities para la iteración
maxiter = 100
tol = 1e-10

# Nueva Variable 
U = np.zeros((N+1, 4))
U[0,:] = U0

for n in range(N):

     rx, ry, vx, vy = U[n, :]

     r_n = np.sqrt(rx**2 + ry**2)

     F_n = np.array([vx, vy, -rx/r_n**3, -ry/r_n**3])

 
     U_inicial= U[n,:]

     for k in range(maxiter):
          
          xk, yk, vxk, vyk = U_inicial
          rk = np.sqrt(xk**2+yk**2)

          F_siguiente = np.array([vxk, vyk, -xk/rk**3, -yk/rk**3])


          # valor de U
          U_nueva = U[n,:] + (Delta_t/2)*(F_n + F_siguiente)

    
          if np.linalg.norm(U_nueva - U_inicial)<tol:
               break

          U_inicial=U_nueva

     U[n+1,:] = U_nueva


valores_x_CN = U[:, 0]
valores_y_CN = U[:, 1]

plt.plot(valores_x_CN, valores_y_CN)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita al problema de Kepler con el método de Crank Nicholson.")
plt.axhline(0, color='black', linewidth=0.8)  # eje x en y=0
plt.axvline(0, color='black', linewidth=0.8)  # eje y en x=0
plt.axis('equal')
plt.show()



###################################
###     Runge Kutta Orden 4     ###
###################################

# Guardo varias soluciones con distintos nombres
U_RK = np.zeros((N+1, 4))

U_RK[0,:] = U0

# Esquema
for n in range(N):
 
 k1 = F(U_RK[n, :])
 k2 = F(U_RK[n, :] + (Delta_t/2)*k1)
 k3 = F(U_RK[n, :] + (Delta_t/2)*k2)
 k4 = F(U_RK[n, :] + Delta_t*k3)

 U_RK[n+1, :] = U_RK[n, :]  + (Delta_t/6)*(k1 + 2*k2 + 2*k3 + k4)


#El mismo plot de siempre
x_rk4 = U_RK[:, 0]
y_rk4 = U_RK[:, 1]



plt.plot(x_rk4, y_rk4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita. Metodo RK-4 orden.")
plt.axhline(0, color='black', linewidth=0.8)  # eje x en y=0
plt.axvline(0, color='black', linewidth=0.8)  # eje y en x=0
plt.axis('equal')
plt.show()


#############################
###     Comparaciones     ###
#############################

##Comparacion RK4 vs Crank-Nicolson vs Euler
plt.figure(figsize=(6,6))
plt.plot(x_euler, y_euler, 'b-', label="Euler")
plt.plot(x_rk4, y_rk4, 'r-', label="RK4")
plt.plot(valores_x_CN, valores_y_CN, 'g-', label ="CrNi")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación Euler vs RK4 vs CN")
plt.legend()
plt.grid(True)
plt.show()

##Comparación RK4 vs CrNi, en el plot completo se ve poquito
plt.figure(figsize=(6,6))
plt.plot(x_rk4, y_rk4, 'r-', label="RK4")
plt.plot(valores_x_CN, valores_y_CN, 'g-', label ="CrNi")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación RK4 vs CN")
plt.legend()
plt.grid(True)
plt.show()

## Resta RK-CN, para ver variabilidad
plt.figure(figsize=(6,6))
plt.plot(x_rk4-valores_x_CN, y_rk4-valores_y_CN, 'r-', label="RK4-CN")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title("resta RK4 - CN")
plt.legend()
plt.grid(True)
plt.show()


