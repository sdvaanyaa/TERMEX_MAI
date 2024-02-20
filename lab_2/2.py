import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CONST VARIABLES
DOT_COLOR = "black"
SPRING_COLOR = "green"
CIRCLE_COLOR = "blue"
CENTRE_COLOR = "white"
CENTRE_OUTLINE_COLOR = "blue"
LINE_COLOR = "black"

A_POINT_SIZE = 4
B_POINT_SIZE = 8

R1 = 5  # радиус большого круга
R2 = 4  # а это маленького

T = 1000

CHART_SIZE = 20

# Координаты
OX1 = -15
OX2 = 15

OY = -R1

n = 10

CENTRE_X = 0
CENTRE_Y = 0

def rot2D(X, Y, Phi):
    RotX = X * np.cos(Phi) - Y * np.sin(Phi)
    RotY = X * np.sin(Phi) + Y * np.cos(Phi)
    return RotX, RotY


def animation(i):
    Centre.set_data(CENTRE_X + vel[i], CENTRE_Y)

    C1.set_data(CX1 + vel[i], CY1)
    C2.set_data(CX2 + vel[i], CY2)

    A.set_data(Ax[i], Ay[i])
    B.set_data(Bx[i], By[i])

    horizont = Bx[i] - Ax[i]
    vertical = By[i] - Ay[i]

    l = np.sqrt(horizont ** 2 + vertical ** 2)
    g = np.pi + np.arctan2(vertical, horizont)

    Rx, Ry = rot2D(SpringX * l, SpringY, g)
    Spring.set_data(Rx + Bx[i], Ry + By[i])

    return [C1, C2, Centre, Spring, A, B]


# Инициализируем диаграмму
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[-CHART_SIZE, CHART_SIZE], ylim=[-CHART_SIZE, CHART_SIZE])

t = np.linspace(0, 10, T)

phi = np.sin(t)
psi = np.cos(t)

vel = R1 * psi
angle = np.linspace(0, 6.28, T) # 2 Pi


# Уравнения движения для окружностей

CX1 = R1 * np.sin(angle) + CENTRE_X
CY1 = R1 * np.cos(angle) + CENTRE_Y

CX2 = R2 * np.sin(angle) + CENTRE_X
CY2 = R2 * np.cos(angle) + CENTRE_Y

BR = (R1 - R2) / 2 + R2

# Уравнения движения точки
Ax = R1 * np.sin(psi) + vel + CENTRE_X
Ay = R1 * np.cos(psi) + CENTRE_Y

Bx = BR * np.sin(phi) + vel + CENTRE_X
By = BR * np.cos(phi) + CENTRE_Y

b = 1 / (n - 2)
sh = 0.5  # 

SpringX = np.zeros(n)
SpringY = np.zeros(n)

SpringX[0] = 0
SpringX[n - 1] = 1

SpringY[0] = 0
SpringY[n - 1] = 0

for i in range(n - 2):
    SpringX[i + 1] = b * (i + 1) - b / 2
    SpringY[i + 1] = sh * (-1) ** i

horizont = Bx[i] - Ax[i]
vertical = By[i] - Ay[i]

l = np.sqrt(horizont ** 2 + vertical ** 2)
g = np.pi + np.arctan2(vertical, horizont)

Rx, Ry = rot2D(SpringX * l, SpringY, g)

# Инициализация объекта
Centre = ax.plot(CENTRE_X + vel[0], CENTRE_Y, CENTRE_COLOR, marker='o', ms=10, mec=CENTRE_OUTLINE_COLOR)[0]

Line = ax.plot([OX1, OX2], [OY, OY], LINE_COLOR)

C1 = ax.plot(CX1 + vel[0], CY1, color=CIRCLE_COLOR)[0]
C2 = ax.plot(CX2 + vel[0], CY2, color=CIRCLE_COLOR)[0]

Spring = ax.plot(Rx + Bx[0], Ry + By[0], SPRING_COLOR)[0]

A = ax.plot(Ax[0], Ay[0], DOT_COLOR, marker='o', ms=A_POINT_SIZE)[0]
B = ax.plot(Bx[0], Ay[0], DOT_COLOR, marker='o', ms=B_POINT_SIZE)[0]

a = FuncAnimation(fig, animation, frames=T, interval=10)

plt.show()