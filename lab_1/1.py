import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

# Определение символьной переменной
t = sp.Symbol('t')

# Определение параметрических уравнений движущейся точки
r = 1 + sp.sin(t)
phi = t
x = r * sp.cos(phi)
y = r * sp.sin(phi)

# Вычисление производных по времени
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)

# Определение центра вращения
cx = x - Vy * ((Vx * Vx + Vy * Vy) / (Vx * Wy - Wx * Vy))
cy = y + Vx * ((Vx * Vx + Vy * Vy) / (Vx * Wy - Wx * Vy))

# Преобразование символьных выражений в функции для численного вычисления
F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)
F_Wx = sp.lambdify(t, Wx)
F_Wy = sp.lambdify(t, Wy)
F_cx = sp.lambdify(t, cx)
F_cy = sp.lambdify(t, cy)

# Генерация временных значений
t = np.linspace(0, 10, 5000)
x = F_x(t)
y = F_y(t)
Vx = F_Vx(t)
Vy = F_Vy(t)
Wx = F_Wx(t)
Wy = F_Wy(t)
cx = F_cx(t)
cy = F_cy(t)
Alpha_V = np.arctan2(Vy, Vx)
Alpha_W = np.arctan2(Wy, Wx)

# Параметры для анимации
k_V = 0.1
k_W = 0.01
a = 0.1
b = 0.03
x_arr = np.array([-a, 0, -a])
y_arr = np.array([b, 0, -b])

# Функция для поворота точек вокруг начала координат
def Rot2D(X, Y, Alpha):
    RotX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RotY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RotX, RotY

# Создание графика
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim=[-2, 2], ylim=[-1, 3])
ax.plot(x, y)

# Инициализация элементов анимации
V_line = ax.plot([x[0], x[0] + k_V * Vx[0]], [y[0], y[0] + k_V * Vy[0]], color=[1, 0, 0])[0]
Rot_Vx, Rot_Vy = Rot2D(x_arr, y_arr, Alpha_V[0])
V_arr = ax.plot(x[0] + k_V * Vx[0] + Rot_Vx, y[0] + k_V * Vy[0] + Rot_Vy, color=[1, 0, 0])[0]

W_line = ax.plot([x[0], x[0] + k_W * Vx[0]], [y[0], y[0] + k_W * Vy[0]], color=[0, 0, 1])[0]
Rot_Wx, Rot_Wy = Rot2D(x_arr, y_arr, Alpha_W[0])
W_arr = ax.plot(x[0] + k_W * Wx[0] + Rot_Wx, y[0] + k_W * Wy[0] + Rot_Wy, color=[0, 0, 1])[0]

c_line = ax.plot([x[0], cx[0]], [y[0], cy[0]], color=[0, 1, 0])[0]

P = ax.plot(x[0], y[0], marker='o')[0]
c = ax.plot(cx[0], cy[0], marker='o')[0]

# Функция для обновления анимации на каждом кадре
def kadr(i):
    P.set_data(x[i], y[i])
    c.set_data(cx[i], cy[i])

    V_line.set_data([x[i], x[i] + k_V * Vx[i]], [y[i], y[i] + k_V * Vy[i]])
    Rot_Vx, Rot_Vy = Rot2D(x_arr, y_arr, Alpha_V[i])
    V_arr.set_data(x[i] + k_V * Vx[i] + Rot_Vx, y[i] + k_V * Vy[i] + Rot_Vy)

    W_line.set_data([x[i], x[i] + k_W * Wx[i]], [y[i], y[i] + k_W * Wy[i]])
    Rot_Wx, Rot_Wy = Rot2D(x_arr, y_arr, Alpha_W[i])
    W_arr.set_data(x[i] + k_W * Wx[i] + Rot_Wx, y[i] + k_W * Wy[i] + Rot_Wy)

    c_line.set_data([x[i], cx[i]], [y[i], cy[i]])

    return [P, c, V_line, V_arr, W_line, W_arr, c_line]

# Создание объекта анимации
kino = FuncAnimation(fig, kadr, frames=len(t), interval=30)

# Отображение анимации
plt.show()
