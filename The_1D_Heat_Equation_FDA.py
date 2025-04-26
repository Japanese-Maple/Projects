import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters

n = 300
m = 1000
l1 = 0
l2 = 3
dx = (l2-l1)/(n+1)
dt = dx**2/4

x_j = np.linspace(l1+dx,l2-dx,n)
t_m = np.linspace(0,m*dt,m)

def A(m):
    a = np.zeros((m,m), dtype=int)
    np.fill_diagonal(a, 2)
    np.fill_diagonal(a[:-1,1:], -1)
    np.fill_diagonal(a[1:,:-1], -1)
    return a
    
# Initial condition

def f(x):
    return (l2-x)*x*(np.cos(3*np.pi*x)**2 + 1)

v = np.zeros((m, n))
v[0, :] = f(x_j)  # Initial condition
for k in range(1, m):
    v[k, :] = np.dot(np.identity(n) - (dt / dx**2) * A(n), v[k - 1, :])
    

fig, HE = plt.subplots()
x_bc = np.concatenate(([l1], x_j, [l2]))
v_bc = np.concatenate(([0], v[0, :], [0]))
line, = HE.plot(x_bc, v_bc, label=f't=0.00', zorder = 2)
IS = HE.plot(x_bc, np.concatenate(([0], f(x_j), [0])), linestyle = 'dashed', zorder = 1)

HE.set_xlim(l1, l2)
HE.set_ylim(0, max(f(x_j) + 0.3))
HE.set_aspect(aspect='equal', adjustable='box')
HE.set_xlabel('x')
HE.set_ylabel('v(x,t)')
HE.set_title('The Heat Equation')
HE.legend()

# Update function for animation
def update(frame):
    v_bc = np.concatenate(([0], v[frame, :], [0]))
    line.set_ydata(v_bc)       # Update only the y-data of the line
    line.set_label(f't={frame * dt:.5f}')  # Update legend label
    line.set_zorder(2)
    HE.legend()       
    return line,

ani = FuncAnimation(fig, update, frames=range(0, 150, 1), blit=True)
ani.save("heat_equation.mp4", writer="ffmpeg", fps=30, dpi=300)

plt.show()





