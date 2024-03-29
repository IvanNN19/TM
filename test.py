import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

x = np.cos(0.2*t) + 3*np.sin(1.8*t)
phi = np.cos(t)#2*np.sin(1.7*t) + 5*np.cos(1.2*t)

SprX_0 = 4
BoxX = 6
BoxY = 2
WheelR = 0.5
l = 5

X_A = SprX_0 + x + BoxX/2
Y_A = 2*WheelR + BoxY/2
X_B = X_A + l*np.sin(phi)
Y_B = Y_A + l*np.cos(phi)

X_Box = np.array([-BoxX/2, BoxX/2, BoxX/2, -BoxX/2, -BoxX/2])
Y_Box = np.array([BoxY/2, BoxY/2, -BoxY/2, -BoxY/2, BoxY/2])

psi = np.linspace(0, 2*math.pi, 20)
X_Wheel = WheelR*np.cos(psi)
Y_Wheel = WheelR*np.sin(psi)

X_Ground = [0, 0, 15]
Y_Ground = [7, 0, 0]

X_C1 = SprX_0 + x + BoxX/5
Y_C1 = WheelR
X_C2 = SprX_0 + x + 4*BoxX/5
Y_C2 = WheelR

K = 19
Sh = 0.4
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0
for i in range(K-2):
    X_Spr[i+1] = b*((i+1) - 1/2)
    Y_Spr[i+1] = Sh*(-1)**i

L_Spr = SprX_0 + x

Nv = 3
R1 = 0.2
R2 = 1
thetta = np.linspace(0, Nv*2*math.pi-phi[0], 100)
X_SpiralSpr = -(R1 + thetta*(R2-R1)/thetta[-1])*np.sin(thetta)
Y_SpiralSpr = (R1 + thetta*(R2-R1)/thetta[-1])*np.cos(thetta)

fig = plt.figure(figsize=[15, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[0, 12], ylim=[-4, 10])

ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)

Drawed_Box = ax.plot(X_Box+X_A[0], Y_Box+Y_A)[0]
Drawed_Wheel1 = ax.plot(X_Wheel+X_C1[0], Y_Wheel+Y_C1)[0]
Drawed_Wheel2 = ax.plot(X_Wheel+X_C2[0], Y_Wheel+Y_C2)[0]

AB = ax.plot([X_A[0], X_B[0]], [Y_A, Y_B[0]])[0]
Point_A = ax.plot(X_A[0], Y_A, marker='o')[0]
Point_B = ax.plot(X_B[0], Y_B[0], marker='o', markersize=20)[0]

Drawed_Spring = ax.plot(X_Spr*L_Spr[0], Y_Spr+Y_A)[0]
Drawed_Spiral_spring = ax.plot(X_SpiralSpr+X_A[0], Y_SpiralSpr+Y_A)[0]

def anima(i):
    Drawed_Box.set_data(X_Box+X_A[i], Y_Box+Y_A)
    Drawed_Wheel1.set_data(X_Wheel+X_C1[i], Y_Wheel+Y_C1)
    Drawed_Wheel2.set_data(X_Wheel+X_C2[i], Y_Wheel+Y_C2)
    AB.set_data([X_A[i], X_B[i]], [Y_A, Y_B[i]])
    Point_A.set_data(X_A[i], Y_A)
    Point_B.set_data(X_B[i], Y_B[i])
    Drawed_Spring.set_data(X_Spr*L_Spr[i], Y_Spr+Y_A)
    thetta = np.linspace(0, Nv * 2 * math.pi - phi[i], 100)
    X_SpiralSpr = -(R1 + thetta * (R2 - R1) / thetta[-1]) * np.sin(thetta)
    Y_SpiralSpr = (R1 + thetta * (R2 - R1) / thetta[-1]) * np.cos(thetta)
    Drawed_Spiral_spring.set_data(X_SpiralSpr+X_A[i], Y_SpiralSpr+Y_A)
    return [Drawed_Box, Drawed_Wheel1, Drawed_Wheel2, AB, Point_A, Point_B, Drawed_Spring, Drawed_Spiral_spring]

anim = FuncAnimation(fig, anima, frames=len(t), interval=40, repeat=False)

plt.show()