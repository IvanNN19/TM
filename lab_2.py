#17_var
#подвинуть пружину!!!1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

phi = 0.5*np.cos(t) + 5
psi = np.cos(t) + 5

L1 = 3
L2 = 2
L0 = 3
BoxX = 2
BoxY = 1
X_Line = [1, 1, 15]
Y_Line = [10, 0, 0]
K = 8
Sh = 0.4
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0

X_A = BoxY + L0 #fix
Y_A = L2 + L1 + 5

X_M1 = X_A + L1*np.cos(phi)
Y_M1 = Y_A + L1*np.sin(phi)

X_M2 = L2*np.cos(psi)
Y_M2 = L2*np.sin(psi)

X_Box = np.array([-BoxX/2, BoxX/2, BoxX/2, -BoxX/2, -BoxX/2])
Y_Box = np.array([BoxY/2, BoxY/2, -BoxY/2, -BoxY/2, BoxY/2])


for i in range(K-2):
    X_Spr[i+1] = b*((i+1) - 1/2)
    Y_Spr[i+1] = Sh*(-1)**i

L_Spr = X_M1 #len spr

fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[0, 15], ylim=[0, 15])

ax.plot(X_Line, Y_Line, color='black', linewidth=3)

Drawed_L1 = ax.plot([X_A, X_M1[0]], [Y_A, Y_M1[0]], color='blue')[0]
Drawed_L2 = ax.plot([X_M1[0], X_M1[0] + X_M2[0]], [Y_M1[0], Y_M1[0] + Y_M2[0]], color='red')[0]
Drawed_Box = ax.plot(X_Box, Y_Box+Y_M1[0])[0]
Drawed_Spring = ax.plot(X_Spr * L_Spr[0], Y_Spr+Y_M1[0], color='black')[0] #подвинуть пружину!!!1
#Drawed_Spring = ax.plot([X_Box, X_Box + X_Spr * L_Spr[0]], [Y_Box+Y_M1[0], Y_Box + Y_Spr[0]])[0]
Point_A = ax.plot(X_A, Y_A, marker='s')
Point_B = ax.plot(X_M1[0], Y_M1[0], marker='o', markersize=10, color='blue')[0]
Point_C = ax.plot(X_M2[0], Y_M2[0], marker='o', markersize=5, color='red')[0]


def anima(i):
    Drawed_L1.set_data([X_A, X_M1[i]], [Y_A, Y_M1[i]])
    Drawed_L2.set_data([X_M1[i], X_M1[i] + X_M2[i]], [Y_M1[i], Y_M1[i] + Y_M2[i]])
    Drawed_Box.set_data(X_Box, Y_Box+Y_M1[i])
    Point_B.set_data(X_M1[i], Y_M1[i])
    Point_C.set_data(X_M2[i] + X_M1[i], Y_M2[i] + Y_M1[i])
    Drawed_Spring.set_data(X_Spr * L_Spr[i], Y_Spr + Y_M1[i])
    #Drawed_Spring.set_data([X_Box, X_Spr * L_Spr[i]], [Y_Box+Y_M1[i], Y_Box + Y_Spr[i]])
    return Drawed_L1, Point_B, Point_C, Drawed_L2, Drawed_Box, Drawed_Spring

anim = FuncAnimation(fig, anima, frames=1000, interval=10)

plt.show()



