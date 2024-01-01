import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def anime(i):
    P.set_data(X[i], Y[i]) #меняем координаты

    VVec.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    RVecArrowX, RVecArrowY = rotation2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VVecArrow.set_data(RVecArrowX + X[i] + VX[i], RVecArrowY + Y[i] + VY[i])

    AVec.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    RVecArrowX_A, RVecArrowY_A = rotation2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AVecArrow.set_data(RVecArrowX_A + X[i] + AX[i], RVecArrowY_A + Y[i] + AY[i])

    RVec.set_data([0, X[i]], [0, Y[i]])
    RVecArrowX_R, RVecArrowY_R = rotation2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RVecArrow.set_data(RVecArrowX_R + X[i], RVecArrowY_R + Y[i])

    NVec.set_data([X[i], X[i] + NX[i]], [Y[i], Y[i] + NY[i]])
    RVecArrowX_N, RVecArrowY_N = rotation2D(ArrowX, ArrowY, math.atan2(NY[i], NX[i]))
    NVecArrow.set_data(RVecArrowX_N + X[i] + NX[i], RVecArrowY_N + Y[i] + NY[i])

    return P, VVec, VVecArrow, AVec, AVecArrow, RVec, RVecArrow, NVec, NVecArrow

def rotation2D(x, y, a):
    Rx = x * np.cos(a) - y * np.sin(a)
    Ry = x * np.sin(a) + y * np.cos(a)
    return Rx, Ry

T = np.linspace(0, 2, 200)
t = sp.Symbol('t')

r = 2 + sp.sin(6*t)
phi = 7*t + 1.2*sp.cos(6*t)

X = np.zeros_like(T)
Y = np.zeros_like(T)
x = r * sp.cos(phi)
y = r * sp.sin(phi)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
NX = np.zeros_like(T)
NY = np.zeros_like(T)

v = (Vx ** 2 + Vy ** 2) ** 0.5
w = (Ax ** 2 + Ay ** 2) ** 0.5
Wtan = sp.diff(v, t)
Wnorm = (w ** 2 - Wtan ** 2) ** 0.5
R = v*v/Wnorm

WtanX = Vx / v * Wtan
WtanY = Vy / v * Wtan
WnormX = Ax - WtanX
WnormY = Ay - WtanY

Wnorm = (WnormX ** 2 + WnormY ** 2) ** 0.5
Nx = WnormX / Wnorm
Ny = WnormY / Wnorm
Rx = Nx * R
Ry = Ny * R

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    NX[i] = sp.Subs(Rx, t, T[i])
    NY[i] = sp.Subs(Ry, t, T[i])

fig = plt.figure() #область отрисовки
ax1 = fig.add_subplot(1, 1, 1) #поле для отрисовки
ax1.axis('equal')#равный масштаб
ax1.set(xlim = [-5, 5], ylim = [-5, 5]) #размер области
ax1.plot(X, Y) #где отрисовываем, откуда берем точки
P, = ax1.plot(X[0], Y[0], marker = 'o') #объект точки

ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])

VVec, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
RVecArrowX, RVecArrowY = rotation2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VVecArrow, = ax1.plot(RVecArrowX + VX[0] + X[0], RVecArrowY + VY[0] + Y[0], 'r')

RVec, = ax1.plot([0, X[0]], [0, Y[0]], 'b')
RVecArrowX_R, RVecArrowY_R = rotation2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RVecArrow, = ax1.plot(RVecArrowX_R + VX[0] + X[0], RVecArrowY_R + VY[0] + Y[0], 'b')

AVec, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'g')
RVecArrowX_A, RVecArrowY_A = rotation2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AVecArrow, = ax1.plot(RVecArrowX_A + X[0], RVecArrowY_A + Y[0], 'g')

NVec, = ax1.plot([X[0], X[0] + NX[0]], [Y[0], Y[0] + NY[0]], 'yellow')
RVecArrowX_N, RVecArrowY_N = rotation2D(ArrowX, ArrowY, math.atan2(NY[0], NX[0]))
NVecArrow, = ax1.plot(RVecArrowX_N + X[0] + NX[0], RVecArrowY_N + Y[0] + NY[0], 'yellow')

anim = FuncAnimation(fig, anime, frames = 200, interval = 50, blit = True, repeat=False)
plt.show()
