import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math
from scipy.integrate import odeint


def odesys(y, t, m, c, L1, L2, g):  # y[3] - psi, y[2] - phi
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 2 * L1
    a12 = L2 * np.cos(y[1] - y[0])
    a21 = L1 * np.cos(y[1] - y[0])
    a22 = L2

    b1 = L2 * (y[3] ** 2) * np.sin(y[1] - y[0]) - ((2 * g + (c * L1 / m) * np.cos(y[0])) * np.sin(y[0]))
    b2 = (-1) * L1 * np.sin(y[1] - y[0]) - g * np.sin(y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy


Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

m = 10
c = 10
L1 = 5
L2 = 5
L0 = 5
g = 9.81

phi0 = 0
psi0 = 0
dphi0 = 0
dpsi0 = (np.pi / 5)
y0 = [phi0, psi0, dphi0, dpsi0]

Y = odeint(odesys, y0, t, (m, c, L1, L2, g))

phi = Y[:, 0]
psi = Y[:, 1]
dphi = Y[:, 2]
dpsi = Y[:, 3]
ddphi = [odesys(y, t, m, c, L1, L2, g)[2] for y, t in zip(Y, t)]
ddpsi = [odesys(y, t, m, c, L1, L2, g)[3] for y, t in zip(Y, t)]

RN = m * (g * np.cos(psi) - L1 * (ddphi * np.sin(psi - phi) - dphi ** 2 * np.cos(psi - phi)) + L2 * dpsi ** 2)

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, psi, color='Red')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, RN, color='Black')
ax_for_graphs.set_title("RN(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

BoxX = 2
BoxY = 1
X_Line = [1, 1, 15]
Y_Line = [10, 0, 0]
K = 8
Sh = 0.4
b = 1 / (K - 2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
psi += 4.73;
phi += 4.73

X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K - 1] = 1
Y_Spr[K - 1] = 0

X_A = BoxY + L0  # fix
Y_A = L2 + L1 + 2

X_M1 = X_A + L1 * np.cos(phi)
Y_M1 = Y_A + L1 * np.sin(phi)

X_M2 = L2 * np.cos(psi)
Y_M2 = L2 * np.sin(psi)

X_Box = np.array([-BoxX / 2, BoxX / 2, BoxX / 2, -BoxX / 2, -BoxX / 2])
Y_Box = np.array([BoxY / 2, BoxY / 2, -BoxY / 2, -BoxY / 2, BoxY / 2])

for i in range(K - 2):
    X_Spr[i + 1] = b * ((i + 1) - 1 / 2)
    Y_Spr[i + 1] = Sh * (-1) ** i

L_Spr = X_M1  # len spr

fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[0, 25], ylim=[0, 15])

ax.plot(X_Line, Y_Line, color='black', linewidth=3)

Drawed_L1 = ax.plot([X_A, X_M1[0]], [Y_A, Y_M1[0]], color='blue')[0]
Drawed_L2 = ax.plot([X_M1[0], X_M1[0] + X_M2[0]], [Y_M1[0], Y_M1[0] + Y_M2[0]], color='red')[0]
Drawed_Box = ax.plot(X_Box, Y_Box + Y_M1[0])[0]
Drawed_Spring = ax.plot(X_Spr * L_Spr[0], Y_Spr + Y_M1[0], color='black')[0]  # подвинуть пружину!!!1
# Drawed_Spring = ax.plot([X_Box, X_Box + X_Spr * L_Spr[0]], [Y_Box+Y_M1[0], Y_Box + Y_Spr[0]])[0]
Point_A = ax.plot(X_A, Y_A, marker='s')
Point_B = ax.plot(X_M1[0], Y_M1[0], marker='o', markersize=10, color='blue')[0]
Point_C = ax.plot(X_M2[0], Y_M2[0], marker='o', markersize=5, color='red')[0]


def anima(i):
    Drawed_L1.set_data([X_A, X_M1[i]], [Y_A, Y_M1[i]])
    Drawed_L2.set_data([X_M1[i], X_M1[i] + X_M2[i]], [Y_M1[i], Y_M1[i] + Y_M2[i]])
    Drawed_Box.set_data(X_Box, Y_Box + Y_M1[i])
    Point_B.set_data(X_M1[i], Y_M1[i])
    Point_C.set_data(X_M2[i] + X_M1[i], Y_M2[i] + Y_M1[i])
    Drawed_Spring.set_data(X_Spr * L_Spr[i], Y_Spr + Y_M1[i])
    # Drawed_Spring.set_data([X_Box, X_Spr * L_Spr[i]], [Y_Box+Y_M1[i], Y_Box + Y_Spr[i]])
    return Drawed_L1, Point_B, Point_C, Drawed_L2, Drawed_Box, Drawed_Spring


anim = FuncAnimation(fig, anima, frames=1000, interval=10)

plt.show()
